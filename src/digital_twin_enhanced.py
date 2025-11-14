"""
Enhanced Digital Twin - Professional Real-time 3D Mapping System
Creates a smooth, accurate, and professional 3D digital twin of the environment
with advanced object detection, tracking, and visualization.
"""
import time
import argparse
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import deque

from capture import CameraCapture
from depth.midas import MiDaSDepth
from pointcloud import depth_to_pointcloud, downsample_pcd
from detection.yolo import YOLOv8Detector
from utils import load_intrinsics, Timer
from mapping import PointCloudMap
from visualization import depth_to_colormap, draw_detections
from digital_twin import COCO_CLASSES, Object3D, ObjectTracker3D, project_2d_to_3d


def create_object_visualization(objects: List[Object3D]) -> Tuple[o3d.geometry.PointCloud, List[o3d.geometry.LineSet]]:
    """
    Create professional visualization for objects in 3D space.
    Returns point cloud of object centers and linesets for 3D markers.
    """
    if len(objects) == 0:
        return o3d.geometry.PointCloud(), []
    
    # Professional color map for different object types
    colors_map = {
        'person': [1.0, 0.2, 0.2],           # Bright Red
        'chair': [0.2, 1.0, 0.2],            # Bright Green
        'laptop': [0.2, 0.4, 1.0],          # Bright Blue
        'door': [1.0, 0.8, 0.0],             # Gold
        'table': [1.0, 0.0, 1.0],            # Magenta
        'couch': [0.0, 1.0, 1.0],            # Cyan
        'bed': [0.8, 0.4, 0.8],              # Purple
        'tv': [0.4, 0.4, 0.4],               # Gray
        'bottle': [0.0, 0.8, 0.4],           # Teal
        'cup': [0.6, 0.3, 0.0],              # Brown
        'book': [0.9, 0.7, 0.3],             # Tan
        'cell phone': [0.5, 0.5, 0.5],       # Silver
    }
    
    points = []
    colors = []
    linesets = []
    
    for obj in objects:
        # Get color for object type, default to white
        color = colors_map.get(obj.class_name.lower(), [1.0, 1.0, 1.0])
        
        # Validate 3D position
        if np.any(np.isnan(obj.center_3d)) or np.any(np.isinf(obj.center_3d)):
            continue
            
        points.append(obj.center_3d)
        colors.append(color)
        
        # Create professional 3D marker (crosshair + sphere approximation)
        marker_size = 0.12  # 12cm marker for better visibility
        center = obj.center_3d
        
        # Create crosshair marker points
        marker_points = [
            center + [marker_size, 0, 0],      # +X
            center + [-marker_size, 0, 0],      # -X
            center + [0, marker_size, 0],       # +Y
            center + [0, -marker_size, 0],      # -Y
            center + [0, 0, marker_size],       # +Z
            center + [0, 0, -marker_size],      # -Z
        ]
        
        # Create lines for 3D crosshair
        lines = [
            [0, 1],  # X-axis
            [2, 3],  # Y-axis
            [4, 5],  # Z-axis
        ]
        
        line_set = o3d.geometry.LineSet()
        all_points = [center] + marker_points
        line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
        linesets.append(line_set)
    
    if len(points) == 0:
        return o3d.geometry.PointCloud(), []
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    return pcd, linesets


def draw_professional_overlay(img: np.ndarray, stats: Dict, fps: float, 
                              tracked_objects: List[Object3D], frame_count: int) -> np.ndarray:
    """
    Draw professional status overlay on image with clear, readable text.
    """
    overlay = img.copy()
    h, w = overlay.shape[:2]
    
    # Semi-transparent background for text readability
    overlay_bg = overlay.copy()
    cv2.rectangle(overlay_bg, (0, 0), (w, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bg, 0.6, overlay, 0.4, 0, overlay)
    
    # Professional font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_large = 0.8
    font_scale_medium = 0.6
    font_scale_small = 0.5
    thickness_large = 2
    thickness_medium = 1
    
    y_pos = 25
    
    # Title
    title = "DIGITAL TWIN SYSTEM - LIVE"
    cv2.putText(overlay, title, (10, y_pos), font, font_scale_large, 
               (0, 255, 255), thickness_large)
    y_pos += 30
    
    # Performance metrics
    fps_color = (0, 255, 0) if fps >= 15 else (0, 165, 255) if fps >= 10 else (0, 0, 255)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(overlay, fps_text, (10, y_pos), font, font_scale_medium, 
               fps_color, thickness_medium)
    
    # Object count
    obj_text = f"Objects: {len(tracked_objects)}"
    text_size = cv2.getTextSize(obj_text, font, font_scale_medium, thickness_medium)[0]
    cv2.putText(overlay, obj_text, (w - text_size[0] - 10, y_pos), font, 
               font_scale_medium, (0, 255, 0), thickness_medium)
    y_pos += 25
    
    # Map statistics
    map_text = f"Map: {stats['points']:,} points | {stats['frames']} frames"
    cv2.putText(overlay, map_text, (10, y_pos), font, font_scale_small, 
               (255, 255, 255), thickness_medium)
    y_pos += 20
    
    # Frame counter
    frame_text = f"Frame: {frame_count}"
    cv2.putText(overlay, frame_text, (10, y_pos), font, font_scale_small, 
               (200, 200, 200), thickness_medium)
    
    # Top tracked objects (right side)
    if len(tracked_objects) > 0:
        y_obj = 25
        max_objects = 5
        for i, obj in enumerate(tracked_objects[:max_objects]):
            obj_name = obj.class_name.upper()
            distance = np.linalg.norm(obj.center_3d)
            obj_text = f"{i+1}. {obj_name} ({distance:.1f}m)"
            text_size = cv2.getTextSize(obj_text, font, font_scale_small, thickness_medium)[0]
            cv2.putText(overlay, obj_text, (w - text_size[0] - 10, y_obj + 50 + i * 18), 
                       font, font_scale_small, (255, 255, 0), thickness_medium)
    
    return overlay


def main(args):
    """Main function for Enhanced Digital Twin system."""
    try:
        cfg, K = load_intrinsics(args.intrinsics)
    except Exception as e:
        print(f"ERROR: Failed to load camera intrinsics: {e}")
        print("Please check the intrinsics file path and format.")
        return

    print("\n" + "="*80)
    print(" " * 20 + "ENHANCED DIGITAL TWIN SYSTEM")
    print(" " * 15 + "Professional Real-time 3D Mapping & Object Detection")
    print("="*80)
    print("\nInitializing components...")
    
    # Initialize components with error handling
    try:
        cam = CameraCapture(cam_id=args.cam, width=cfg['image_width'], height=cfg['image_height'])
        print("✓ Camera initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize camera: {e}")
        return
    
    try:
        depth_model = MiDaSDepth(model_type=args.depth_model)
        print(f"✓ Depth model loaded: {args.depth_model}")
    except Exception as e:
        print(f"ERROR: Failed to load depth model: {e}")
        return
    
    try:
        detector = YOLOv8Detector(model=args.yolo_model)
        print(f"✓ Object detector loaded: {args.yolo_model}")
    except Exception as e:
        print(f"ERROR: Failed to load object detector: {e}")
        return
    
    # Point cloud map for accumulation
    pcd_map = PointCloudMap(voxel_size=args.voxel_size, max_points=args.max_points)
    print("✓ Point cloud map initialized")
    
    # Object tracker
    object_tracker = ObjectTracker3D(max_distance=args.track_distance, max_age=args.track_age)
    print("✓ Object tracker initialized")
    
    # Visualization windows
    cv2.namedWindow('Live View (RGB + Detections)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    print("✓ Visualization windows created")
    
    # 3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Digital Twin - 3D Map with Objects', width=1280, height=720)
    
    # Add geometries
    map_pcd = o3d.geometry.PointCloud()
    objects_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(map_pcd, reset_bounding_box=False)
    vis.add_geometry(objects_pcd, reset_bounding_box=False)
    
    object_linesets = []  # Store linesets for object markers
    
    # Render once to initialize
    vis.poll_events()
    vis.update_renderer()
    print("✓ 3D visualization initialized")
    
    # Performance tracking
    frame_count = 0
    last_map_update = 0
    map_update_interval = args.map_update_freq
    fps_history = deque(maxlen=30)  # Track FPS over last 30 frames
    view_initialized = False
    
    print("\n" + "="*80)
    print("SYSTEM READY - Starting real-time mapping...")
    print("="*80)
    print("\nControls:")
    print("  • Move camera slowly to build the 3D map")
    print("  • Objects are automatically detected and tracked in 3D space")
    print("  • Press 's' - Save current map and objects")
    print("  • Press 'c' - Clear map and start over")
    print("  • Press 'i' - Print detailed object information")
    print("  • Press 'q' or ESC - Quit gracefully")
    print("\n" + "="*80 + "\n")

    try:
        while True:
            t = Timer()
            t.start()
            
            # Capture frame with error handling
            try:
                rgb = cam.read()
            except Exception as e:
                print(f"WARNING: Frame capture failed: {e}")
                continue
            
            # Estimate depth with validation
            try:
                depth = depth_model.predict(rgb)
                # Validate depth map
                if depth is None or depth.size == 0 or np.all(depth <= 0):
                    print("WARNING: Invalid depth map, skipping frame")
                    continue
            except Exception as e:
                print(f"WARNING: Depth estimation failed: {e}")
                continue
            
            # Detect objects
            try:
                detections_2d = detector.detect(rgb)
            except Exception as e:
                print(f"WARNING: Object detection failed: {e}")
                detections_2d = []
            
            # Project detections to 3D with validation
            detections_3d = []
            for det in detections_2d:
                # Filter by confidence
                if det['conf'] < 0.3:  # Minimum confidence threshold
                    continue
                    
                try:
                    center_3d = project_2d_to_3d(det['bbox'], depth, K)
                    if center_3d is not None:
                        # Validate 3D position
                        if (np.all(np.isfinite(center_3d)) and 
                            np.linalg.norm(center_3d) < 10.0 and  # Within 10m
                            center_3d[2] > 0.1):  # Valid depth
                            class_name = COCO_CLASSES.get(det['class'], f'class_{det["class"]}')
                            detections_3d.append({
                                'class_id': det['class'],
                                'class_name': class_name,
                                'center_3d': center_3d,
                                'bbox_2d': det['bbox'],
                                'confidence': det['conf']
                            })
                except Exception as e:
                    continue  # Skip invalid detections
            
            # Update object tracker
            object_tracker.update(detections_3d, frame_count)
            tracked_objects = object_tracker.get_objects()
            
            # Update map periodically (optimized)
            if frame_count - last_map_update >= map_update_interval:
                try:
                    pcd = depth_to_pointcloud(rgb, depth, K)
                    if len(pcd.points) > 0:
                        pcd = downsample_pcd(pcd, voxel_size=args.voxel_size)
                        pcd_map.add_frame(pcd)
                        last_map_update = frame_count
                except Exception as e:
                    print(f"WARNING: Map update failed: {e}")
            
            # Get map statistics
            map_stats = pcd_map.get_stats()
            
            # Calculate FPS
            elapsed = t.elapsed_ms()
            fps = 1000.0 / (elapsed + 1e-9)
            fps_history.append(fps)
            avg_fps = np.mean(fps_history) if len(fps_history) > 0 else fps
            
            # Update visualization
            # 1. RGB with detections and professional overlay
            try:
                display_rgb = draw_detections(rgb.copy(), detections_2d, COCO_CLASSES)
                display_rgb = draw_professional_overlay(display_rgb, map_stats, avg_fps, 
                                                        tracked_objects, frame_count)
                cv2.imshow('Live View (RGB + Detections)', cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"WARNING: RGB display update failed: {e}")
            
            # 2. Depth map with professional styling
            try:
                depth_colored = depth_to_colormap(depth)
                # Add depth scale indicator
                h, w = depth_colored.shape[:2]
                cv2.putText(depth_colored, "DEPTH MAP", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(depth_colored, f"Range: {depth.min():.2f}m - {depth.max():.2f}m", 
                           (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('Depth Map', cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
            except Exception as e:
                print(f"WARNING: Depth display update failed: {e}")
            
            # 3. Update 3D map (optimized - only when changed)
            global_map = pcd_map.get_map()
            if len(global_map.points) > 0:
                try:
                    # Only update if map changed
                    if len(map_pcd.points) != len(global_map.points) or frame_count % 5 == 0:
                        map_pcd.points = global_map.points
                        map_pcd.colors = global_map.colors
                        vis.update_geometry(map_pcd)
                except Exception as e:
                    print(f"WARNING: 3D map update failed: {e}")
            
            # 4. Update object visualization (optimized)
            if len(tracked_objects) > 0:
                try:
                    # Remove old object markers
                    for line_set in object_linesets:
                        try:
                            vis.remove_geometry(line_set, reset_bounding_box=False)
                        except:
                            pass
                    object_linesets.clear()
                    
                    # Create new visualization
                    objects_pcd, new_linesets = create_object_visualization(tracked_objects)
                    
                    # Update object point cloud
                    if len(objects_pcd.points) > 0:
                        vis.remove_geometry(objects_pcd, reset_bounding_box=False)
                        vis.add_geometry(objects_pcd, reset_bounding_box=False)
                    
                    # Add linesets
                    for line_set in new_linesets:
                        vis.add_geometry(line_set, reset_bounding_box=False)
                        object_linesets.append(line_set)
                except Exception as e:
                    print(f"WARNING: Object visualization update failed: {e}")
            
            # Auto-fit view on first frame with map
            if not view_initialized and len(global_map.points) > 100:
                try:
                    view_ctl = vis.get_view_control()
                    center = global_map.get_center()
                    view_ctl.set_lookat(center)
                    view_ctl.set_front([0, 0, -1])
                    view_ctl.set_up([0, -1, 0])
                    
                    # Calculate appropriate zoom
                    bbox = global_map.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()
                    max_extent = max(extent)
                    if max_extent > 0:
                        zoom = 0.6 / max_extent
                        view_ctl.set_zoom(min(zoom, 0.9))
                    
                    view_initialized = True
                    print("✓ 3D view initialized and optimized")
                except Exception as e:
                    print(f"WARNING: View initialization failed: {e}")
            
            # Smooth rendering
            vis.poll_events()
            vis.update_renderer()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n" + "="*80)
                print("Shutting down gracefully...")
                break
            elif key == ord('s'):
                # Save map and objects
                try:
                    timestamp = int(time.time())
                    map_filename = f"digital_twin_map_{timestamp}.ply"
                    pcd_map.save(map_filename)
                    
                    # Save object list with professional formatting
                    obj_filename = f"digital_twin_objects_{timestamp}.txt"
                    with open(obj_filename, 'w') as f:
                        f.write("="*80 + "\n")
                        f.write(" " * 25 + "DIGITAL TWIN - DETECTED OBJECTS\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"Export Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Total Objects: {len(tracked_objects)}\n")
                        f.write(f"Map Points: {map_stats['points']:,}\n")
                        f.write(f"Frames Processed: {map_stats['frames']}\n")
                        f.write("\n" + "-"*80 + "\n\n")
                        
                        for i, obj in enumerate(tracked_objects, 1):
                            distance = np.linalg.norm(obj.center_3d)
                            f.write(f"{i}. {obj.class_name.upper()} (Class ID: {obj.class_id})\n")
                            f.write(f"   Position (x, y, z): ({obj.center_3d[0]:.3f}, {obj.center_3d[1]:.3f}, {obj.center_3d[2]:.3f}) meters\n")
                            f.write(f"   Distance from origin: {distance:.3f} meters\n")
                            f.write(f"   Confidence: {obj.confidence:.3f}\n")
                            f.write(f"   Last seen: frame {obj.last_seen}\n")
                            f.write("\n")
                    
                    print("\n" + "="*80)
                    print("✓ SAVE SUCCESSFUL")
                    print("="*80)
                    print(f"  Map saved: {map_filename}")
                    print(f"  Objects saved: {obj_filename}")
                    print(f"  Objects exported: {len(tracked_objects)}")
                    print("="*80 + "\n")
                except Exception as e:
                    print(f"ERROR: Failed to save: {e}")
            elif key == ord('c'):
                # Clear map and objects
                pcd_map.clear()
                object_tracker.objects.clear()
                # Remove all object markers
                for line_set in object_linesets:
                    try:
                        vis.remove_geometry(line_set, reset_bounding_box=False)
                    except:
                        pass
                object_linesets.clear()
                view_initialized = False
                print("\n" + "="*80)
                print("✓ Map and objects cleared - Ready for new mapping session")
                print("="*80 + "\n")
            elif key == ord('i'):
                # Print detailed object information
                print("\n" + "="*80)
                print(" " * 30 + "TRACKED OBJECTS")
                print("="*80)
                if len(tracked_objects) == 0:
                    print("  No objects currently tracked.")
                else:
                    for i, obj in enumerate(tracked_objects, 1):
                        distance = np.linalg.norm(obj.center_3d)
                        print(f"\n{i}. {obj.class_name.upper()} (Class ID: {obj.class_id})")
                        print(f"   Position: ({obj.center_3d[0]:.3f}, {obj.center_3d[1]:.3f}, {obj.center_3d[2]:.3f}) m")
                        print(f"   Distance: {distance:.3f} m")
                        print(f"   Confidence: {obj.confidence:.3f}")
                        print(f"   Last seen: frame {obj.last_seen}")
                print("\n" + "="*80 + "\n")
            
            frame_count += 1
            
            # Print stats periodically (less verbose)
            if frame_count % 60 == 0:
                print(f"[Frame {frame_count}] Objects: {len(tracked_objects)} | "
                      f"Map: {map_stats['points']:,} pts | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Frames in map: {map_stats['frames']}")

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Interrupted by user - Shutting down gracefully...")
        print("="*80)
    except Exception as e:
        print(f"\n\nERROR: Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Auto-save on exit
        print("\nFinalizing session...")
        try:
            if args.auto_save and 'pcd_map' in locals():
                global_map = pcd_map.get_map()
                if len(global_map.points) > 0:
                    map_filename = "digital_twin_auto_save.ply"
                    pcd_map.save(map_filename)
                    print(f"✓ Auto-saved map to {map_filename}")
                else:
                    print("  No map data to save")
        except Exception as e:
            print(f"  WARNING: Auto-save failed: {e}")
        
        # Cleanup
        try:
            if 'cam' in locals():
                cam.release()
            cv2.destroyAllWindows()
            if 'vis' in locals():
                vis.destroy_window()
        except:
            pass
        
        print("\n" + "="*80)
        print(" " * 25 + "SESSION ENDED SUCCESSFULLY")
        print("="*80)
        try:
            if 'frame_count' in locals():
                print(f"  Total frames processed: {frame_count}")
            if 'pcd_map' in locals():
                print(f"  Final map points: {len(pcd_map.get_map().points):,}")
            if 'object_tracker' in locals():
                tracked_objects = object_tracker.get_objects()
                print(f"  Final tracked objects: {len(tracked_objects)}")
        except:
            pass
        print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced Digital Twin - Real-time 3D mapping with object detection')
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
                       help='Update map every N frames')
    parser.add_argument('--auto_save', action='store_true',
                       help='Auto-save map on exit')
    
    args = parser.parse_args()
    
    # Adjust intrinsics path
    if args.intrinsics.startswith('..'):
        args.intrinsics = 'config/camera_intrinsics.yaml'
    
    main(args)

