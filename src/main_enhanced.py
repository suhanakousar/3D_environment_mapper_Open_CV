"""
Enhanced 3D Mapper with accumulation, export, and multi-view features.
"""
import time
import argparse
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path

from capture import CameraCapture
from depth.midas import MiDaSDepth
from pointcloud import depth_to_pointcloud, downsample_pcd
from detection.yolo import YOLOv8Detector
from utils import load_intrinsics, Timer
from mapping import PointCloudMap
from visualization import create_multi_view, depth_to_colormap, draw_detections


def main(args):
    cfg, K = load_intrinsics(args.intrinsics)

    # Initialize components
    cam = CameraCapture(cam_id=args.cam, width=cfg['image_width'], height=cfg['image_height'])
    depth_model = MiDaSDepth(model_type=args.depth_model)
    
    detector = None
    if args.detect:
        detector = YOLOv8Detector(model=args.yolo_model)
    
    # Point cloud map for accumulation
    pcd_map = None
    if args.accumulate:
        pcd_map = PointCloudMap(voxel_size=args.voxel_size, max_points=args.max_points)
        print("Point cloud accumulation enabled")
    
    # 3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Mapper - Live', width=cfg['image_width'], height=cfg['image_height'])
    pcd_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_geom)
    
    # Set up view control
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, 0, -1])
    view_ctl.set_lookat([0, 0, 2])
    view_ctl.set_up([0, -1, 0])
    view_ctl.set_zoom(0.7)
    
    # Multi-view window (optional)
    multi_view_window = None
    if args.multi_view:
        multi_view_window = "Multi-View (RGB + Depth)"
        cv2.namedWindow(multi_view_window, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    export_count = 0
    
    print("\nControls:")
    print("  - Press 's' to save current point cloud")
    print("  - Press 'm' to save accumulated map (if accumulation enabled)")
    print("  - Press 'c' to clear accumulated map")
    print("  - Press 'q' or ESC to quit")
    print("  - Press SPACE to pause/resume\n")

    try:
        paused = False
        while True:
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s'):  # Save current frame
                if len(pcd_geom.points) > 0:
                    filename = f"pointcloud_frame_{export_count:04d}.ply"
                    o3d.io.write_point_cloud(filename, pcd_geom)
                    print(f"Saved current frame to {filename}")
                    export_count += 1
            elif key == ord('m') and pcd_map:  # Save map
                filename = f"map_accumulated_{export_count:04d}.ply"
                pcd_map.save(filename)
                export_count += 1
            elif key == ord('c') and pcd_map:  # Clear map
                pcd_map.clear()
                print("Accumulated map cleared")
            
            if paused:
                vis.poll_events()
                vis.update_renderer()
                continue
            
            t = Timer()
            t.start()
            
            # Capture frame
            rgb = cam.read()
            depth = depth_model.predict(rgb)
            
            # Create point cloud
            pcd = depth_to_pointcloud(rgb, depth, K)
            pcd = downsample_pcd(pcd, voxel_size=args.voxel_size)
            
            # Accumulate if enabled
            if pcd_map:
                pcd_map.add_frame(pcd)
                # Display accumulated map
                display_pcd = pcd_map.get_map()
            else:
                display_pcd = pcd
            
            # Update 3D visualization
            pcd_geom.points = display_pcd.points
            pcd_geom.colors = display_pcd.colors
            vis.update_geometry(pcd_geom)
            vis.poll_events()
            vis.update_renderer()
            
            # Auto-fit view on first frame
            if frame_count == 0 and len(display_pcd.points) > 0:
                view_ctl = vis.get_view_control()
                view_ctl.set_front([0, 0, -1])
                view_ctl.set_lookat(display_pcd.get_center())
                view_ctl.set_up([0, -1, 0])
                view_ctl.set_zoom(0.7)
            
            # Object detection
            dets = None
            if detector:
                dets = detector.detect(rgb)
                if len(dets) > 0:
                    print(f"Detections: {len(dets)}")
            
            # Multi-view display
            if multi_view_window:
                multi_img = create_multi_view(rgb, depth, dets, 
                                            width=cfg['image_width'], 
                                            height=cfg['image_height'])
                cv2.imshow(multi_view_window, cv2.cvtColor(multi_img, cv2.COLOR_RGB2BGR))
            
            # Print stats
            elapsed = t.elapsed_ms()
            fps = 1000.0 / (elapsed + 1e-9)
            num_points = len(display_pcd.points)
            
            stats_line = f"Frame {frame_count}: {elapsed:.1f}ms ({fps:.1f} FPS), {num_points} points"
            if pcd_map:
                map_stats = pcd_map.get_stats()
                stats_line += f", Map: {map_stats['points']} points from {map_stats['frames']} frames"
            print(stats_line)
            
            frame_count += 1

    except KeyboardInterrupt:
        print('\nExiting...')
    finally:
        # Save accumulated map on exit if enabled
        if pcd_map and len(pcd_map.get_map().points) > 0:
            if args.auto_save:
                filename = "map_auto_save.ply"
                pcd_map.save(filename)
                print(f"Auto-saved map to {filename}")
        
        cam.release()
        if multi_view_window:
            cv2.destroyAllWindows()
        vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced 3D Mapper with accumulation and export')
    parser.add_argument('--cam', type=int, default=0, help='Camera ID')
    parser.add_argument('--intrinsics', type=str, default='config/camera_intrinsics.yaml',
                       help='Camera intrinsics file')
    parser.add_argument('--detect', action='store_true', help='Enable object detection')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt', help='YOLO model file')
    parser.add_argument('--depth_model', type=str, default='DPT_Hybrid',
                       choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                       help='MiDaS depth model type')
    parser.add_argument('--accumulate', action='store_true',
                       help='Accumulate point clouds over time')
    parser.add_argument('--voxel_size', type=float, default=0.03,
                       help='Voxel size for downsampling (meters)')
    parser.add_argument('--max_points', type=int, default=1000000,
                       help='Maximum points in accumulated map')
    parser.add_argument('--multi_view', action='store_true',
                       help='Show multi-view window (RGB + Depth)')
    parser.add_argument('--auto_save', action='store_true',
                       help='Auto-save accumulated map on exit')
    
    args = parser.parse_args()
    
    # Adjust intrinsics path
    if args.intrinsics.startswith('..'):
        args.intrinsics = 'config/camera_intrinsics.yaml'
    
    main(args)

