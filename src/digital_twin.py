"""
Digital Twin System - Real-time 3D mapping with object detection and labeling.
Creates a live 3D model of the environment with labeled objects.
"""
import time
import argparse
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Dict, Tuple

from capture import CameraCapture
from depth.midas import MiDaSDepth
from pointcloud import depth_to_pointcloud, downsample_pcd
from detection.yolo import YOLOv8Detector
from utils import load_intrinsics, Timer
from mapping import PointCloudMap
from visualization import depth_to_colormap, draw_detections


# COCO class names for YOLO
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


class Object3D:
    """Represents a detected object in 3D space."""
    def __init__(self, class_id: int, class_name: str, bbox_2d: np.ndarray, 
                 center_3d: np.ndarray, confidence: float, frame_id: int):
        self.class_id = class_id
        self.class_name = class_name
        self.bbox_2d = bbox_2d  # [x1, y1, x2, y2]
        self.center_3d = center_3d  # [x, y, z]
        self.confidence = confidence
        self.frame_id = frame_id
        self.last_seen = frame_id
        self.bbox_3d = None  # Will be computed if needed
        
    def update(self, center_3d: np.ndarray, frame_id: int):
        """Update object position (simple averaging for now)."""
        # Simple moving average
        alpha = 0.3
        self.center_3d = alpha * center_3d + (1 - alpha) * self.center_3d
        self.last_seen = frame_id


class ObjectTracker3D:
    """Tracks objects in 3D space across frames."""
    def __init__(self, max_distance: float = 0.5, max_age: int = 10):
        self.objects: List[Object3D] = []
        self.max_distance = max_distance  # meters
        self.max_age = max_age  # frames
        self.next_id = 0
        
    def update(self, detections_3d: List[Dict], frame_id: int):
        """
        Update tracked objects with new detections.
        detections_3d: List of dicts with 'class_id', 'class_name', 'center_3d', 'confidence'
        """
        # Remove old objects
        self.objects = [obj for obj in self.objects 
                       if (frame_id - obj.last_seen) < self.max_age]
        
        # Match new detections to existing objects
        matched = set()
        for det in detections_3d:
            center_3d = det['center_3d']
            best_match = None
            best_dist = self.max_distance
            
            for obj in self.objects:
                if obj.class_id == det['class_id']:
                    dist = np.linalg.norm(center_3d - obj.center_3d)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = obj
            
            if best_match:
                best_match.update(center_3d, frame_id)
                matched.add(id(best_match))
            else:
                # New object
                obj = Object3D(
                    det['class_id'],
                    det['class_name'],
                    det.get('bbox_2d', np.array([0, 0, 0, 0])),
                    center_3d,
                    det['confidence'],
                    frame_id
                )
                self.objects.append(obj)
        
    def get_objects(self) -> List[Object3D]:
        """Get all currently tracked objects."""
        return self.objects


def project_2d_to_3d(bbox_2d: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 2D bounding box center to 3D using depth map.
    
    Args:
        bbox_2d: [x1, y1, x2, y2] in pixels
        depth: Depth map (H, W) in meters
        K: Camera intrinsics matrix
        
    Returns:
        3D point [x, y, z] in camera coordinates, or None if invalid
    """
    x1, y1, x2, y2 = bbox_2d.astype(int)
    
    # Get center of bounding box
    cx_2d = (x1 + x2) / 2
    cy_2d = (y1 + y2) / 2
    
    # Sample depth at center (with some averaging)
    h, w = depth.shape
    cx_int = int(np.clip(cx_2d, 0, w - 1))
    cy_int = int(np.clip(cy_2d, 0, h - 1))
    
    # Sample a small region around center
    region_size = 5
    y_min = max(0, cy_int - region_size)
    y_max = min(h, cy_int + region_size)
    x_min = max(0, cx_int - region_size)
    x_max = min(w, cx_int + region_size)
    
    depth_region = depth[y_min:y_max, x_min:x_max]
    valid_depths = depth_region[depth_region > 0.1]
    
    if len(valid_depths) == 0:
        return None
    
    z = np.median(valid_depths)
    
    if z < 0.1 or z > 10.0:  # Invalid depth
        return None
    
    # Project to 3D
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    x = (cx_2d - cx) * z / fx
    y = (cy_2d - cy) * z / fy
    
    return np.array([x, y, z])


def create_object_labels_pcd(objects: List[Object3D]) -> o3d.geometry.PointCloud:
    """Create a point cloud with object centers for visualization."""
    if len(objects) == 0:
        return o3d.geometry.PointCloud()
    
    points = np.array([obj.center_3d for obj in objects])
    colors = np.array([[1.0, 0.0, 0.0] for _ in objects])  # Red for object centers
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def main(args):
    cfg, K = load_intrinsics(args.intrinsics)

    print("\n" + "="*70)
    print("DIGITAL TWIN SYSTEM - Real-time 3D Mapping with Object Detection")
    print("="*70)
    print("Building a live 3D model of your environment...")
    print("="*70 + "\n")

    # Initialize components
    cam = CameraCapture(cam_id=args.cam, width=cfg['image_width'], height=cfg['image_height'])
    depth_model = MiDaSDepth(model_type=args.depth_model)
    detector = YOLOv8Detector(model=args.yolo_model)
    
    # Point cloud map for accumulation
    pcd_map = PointCloudMap(voxel_size=args.voxel_size, max_points=args.max_points)
    
    # Object tracker
    object_tracker = ObjectTracker3D(max_distance=args.track_distance, max_age=args.track_age)
    
    # Visualization windows
    cv2.namedWindow('Live View (RGB + Detections)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    
    # 3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Digital Twin - 3D Map', width=1280, height=720)
    
    # Add geometries
    map_pcd = o3d.geometry.PointCloud()
    objects_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(map_pcd, reset_bounding_box=False)
    vis.add_geometry(objects_pcd, reset_bounding_box=False)
    
    # Render once to initialize
    vis.poll_events()
    vis.update_renderer()
    
    frame_count = 0
    last_map_update = 0
    map_update_interval = args.map_update_freq  # Update map every N frames
    
    print("Controls:")
    print("  Move camera slowly to build the 3D map")
    print("  Press 's' - Save current map and objects")
    print("  Press 'c' - Clear map and start over")
    print("  Press 'q' or ESC - Quit")
    print("\nStarting...\n")

    try:
        while True:
            t = Timer()
            t.start()
            
            # Capture frame
            rgb = cam.read()
            
            # Estimate depth
            depth = depth_model.predict(rgb)
            
            # Detect objects
            detections_2d = detector.detect(rgb)
            
            # Project detections to 3D
            detections_3d = []
            for det in detections_2d:
                center_3d = project_2d_to_3d(det['bbox'], depth, K)
                if center_3d is not None:
                    class_name = COCO_CLASSES.get(det['class'], f'class_{det["class"]}')
                    detections_3d.append({
                        'class_id': det['class'],
                        'class_name': class_name,
                        'center_3d': center_3d,
                        'bbox_2d': det['bbox'],
                        'confidence': det['conf']
                    })
            
            # Update object tracker
            object_tracker.update(detections_3d, frame_count)
            tracked_objects = object_tracker.get_objects()
            
            # Update map periodically (not every frame for performance)
            if frame_count - last_map_update >= map_update_interval:
                pcd = depth_to_pointcloud(rgb, depth, K)
                pcd = downsample_pcd(pcd, voxel_size=args.voxel_size)
                pcd_map.add_frame(pcd)
                last_map_update = frame_count
            
            # Update visualization
            # 1. RGB with detections
            display_rgb = draw_detections(rgb.copy(), detections_2d, COCO_CLASSES)
            
            # Add object count overlay
            obj_count_text = f"Objects detected: {len(tracked_objects)}"
            cv2.putText(display_rgb, obj_count_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            map_stats = pcd_map.get_stats()
            map_text = f"Map: {map_stats['points']} points, {map_stats['frames']} frames"
            cv2.putText(display_rgb, map_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Live View (RGB + Detections)', cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
            
            # 2. Depth map
            depth_colored = depth_to_colormap(depth)
            cv2.imshow('Depth Map', cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
            
            # 3. Update 3D map
            global_map = pcd_map.get_map()
            if len(global_map.points) > 0:
                map_pcd.points = global_map.points
                map_pcd.colors = global_map.colors
                vis.update_geometry(map_pcd)
            
            # 4. Update object markers
            if len(tracked_objects) > 0:
                objects_pcd = create_object_labels_pcd(tracked_objects)
                vis.remove_geometry(objects_pcd, reset_bounding_box=False)
                vis.add_geometry(objects_pcd, reset_bounding_box=False)
            
            # Auto-fit view on first frame
            if frame_count == 0 and len(global_map.points) > 0:
                view_ctl = vis.get_view_control()
                center = global_map.get_center()
                view_ctl.set_lookat(center)
                view_ctl.set_front([0, 0, -1])
                view_ctl.set_up([0, -1, 0])
                view_ctl.set_zoom(0.7)
            
            vis.poll_events()
            vis.update_renderer()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                # Save map
                map_filename = f"digital_twin_map_{int(time.time())}.ply"
                pcd_map.save(map_filename)
                
                # Save object list
                obj_filename = f"digital_twin_objects_{int(time.time())}.txt"
                with open(obj_filename, 'w') as f:
                    f.write("Digital Twin - Detected Objects\n")
                    f.write("="*50 + "\n")
                    for obj in tracked_objects:
                        f.write(f"{obj.class_name} (ID: {obj.class_id})\n")
                        f.write(f"  Position: {obj.center_3d}\n")
                        f.write(f"  Confidence: {obj.confidence:.2f}\n")
                        f.write(f"  Last seen: frame {obj.last_seen}\n")
                        f.write("\n")
                
                print(f"\n✓ Saved map to {map_filename}")
                print(f"✓ Saved objects to {obj_filename}\n")
            elif key == ord('c'):
                pcd_map.clear()
                object_tracker.objects.clear()
                print("Map and objects cleared!\n")
            
            frame_count += 1
            
            # Print stats periodically
            if frame_count % 30 == 0:
                elapsed = t.elapsed_ms()
                fps = 1000.0 / (elapsed + 1e-9)
                print(f"Frame {frame_count}: {len(tracked_objects)} objects tracked | "
                      f"Map: {map_stats['points']} points | {fps:.1f} FPS")

    except KeyboardInterrupt:
        print('\nExiting...')
    finally:
        # Auto-save on exit
        if args.auto_save and len(pcd_map.get_map().points) > 0:
            map_filename = "digital_twin_auto_save.ply"
            pcd_map.save(map_filename)
            print(f"Auto-saved map to {map_filename}")
        
        cam.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("Digital Twin session ended.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Digital Twin - Real-time 3D mapping with object detection')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--intrinsics', type=str, default='config/camera_intrinsics.yaml',
                       help='Camera intrinsics file')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='YOLO model file')
    parser.add_argument('--depth_model', type=str, default='DPT_Hybrid',
                       choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                       help='MiDaS depth model type')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                       help='Voxel size for downsampling (meters)')
    parser.add_argument('--max_points', type=int, default=2000000,
                       help='Maximum points in map')
    parser.add_argument('--track_distance', type=float, default=0.5,
                       help='Max distance to match objects (meters)')
    parser.add_argument('--track_age', type=int, default=30,
                       help='Max frames before removing untracked object')
    parser.add_argument('--map_update_freq', type=int, default=5,
                       help='Update map every N frames (higher = faster but less detailed)')
    parser.add_argument('--auto_save', action='store_true',
                       help='Auto-save map on exit')
    
    args = parser.parse_args()
    
    # Adjust intrinsics path
    if args.intrinsics.startswith('..'):
        args.intrinsics = 'config/camera_intrinsics.yaml'
    
    main(args)

