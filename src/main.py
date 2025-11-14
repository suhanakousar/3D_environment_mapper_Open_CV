import time
import argparse
import numpy as np
import open3d as o3d
import cv2

from capture import CameraCapture
from depth.midas import MiDaSDepth
from pointcloud import depth_to_pointcloud, downsample_pcd
from detection.yolo import YOLOv8Detector
from utils import load_intrinsics, Timer
from visualization import depth_to_colormap, draw_detections


def main(args):
    cfg, K = load_intrinsics(args.intrinsics)

    cam = CameraCapture(cam_id=args.cam, width=cfg['image_width'], height=cfg['image_height'])
    depth_model = MiDaSDepth(model_type='DPT_Hybrid')

    detector = None
    if args.detect:
        detector = YOLOv8Detector(model=args.yolo_model)

    # Create webcam preview window - always show the input
    cv2.namedWindow('Webcam Input', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
    
    # 3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Structure (from Webcam)', width=cfg['image_width'], height=cfg['image_height'])
    
    # Create initial empty point cloud
    pcd_geom = o3d.geometry.PointCloud()
    vis.add_geometry(pcd_geom, reset_bounding_box=False)
    
    # Render once to initialize window
    vis.poll_events()
    vis.update_renderer()
    
    geometry_added = False
    
    frame_count = 0
    
    print("\n" + "="*60)
    print("3D Structure Builder from Webcam - Camera Mode")
    print("="*60)
    print("You will see:")
    print("  1. 'Webcam Input' - Live camera preview (with capture button)")
    print("  2. 'Depth Map' - Estimated depth of captured image")
    print("  3. '3D Structure' - 3D point cloud created from captured image")
    print("\nControls:")
    print("  Press 'c' or SPACE - CAPTURE image and create 3D structure")
    print("  Press 's' - Save current 3D structure to file")
    print("  Press 'q' or ESC - Quit")
    print("="*60 + "\n")
    
    save_count = 0
    captured_frame = None
    captured_rgb = None
    captured_depth = None
    is_captured = False

    try:
        num_points = 0  # Initialize for save check
        while True:
            t = Timer(); t.start()
            
            # Step 1: Get live preview from webcam
            rgb = cam.read()  # RGB uint8
            
            # Display webcam preview with capture button overlay
            display_rgb = rgb.copy()
            
            # Add capture button overlay
            h, w = display_rgb.shape[:2]
            button_y = h - 80
            button_x = w // 2 - 100
            button_w = 200
            button_h = 50
            
            # Draw capture button
            cv2.rectangle(display_rgb, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), 
                         (0, 255, 0), -1)  # Green filled rectangle
            cv2.rectangle(display_rgb, (button_x, button_y), 
                         (button_x + button_w, button_y + button_h), 
                         (255, 255, 255), 2)  # White border
            
            # Add text
            text = "CAPTURE (Press 'c')"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = button_x + (button_w - text_size[0]) // 2
            text_y = button_y + (button_h + text_size[1]) // 2
            cv2.putText(display_rgb, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Add status text
            if is_captured:
                status_text = "Image Captured! Processing 3D structure..."
                cv2.putText(display_rgb, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                status_text = "Live Preview - Press 'c' to capture"
                cv2.putText(display_rgb, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show webcam preview
            cv2.imshow('Webcam Input', cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR))
            
            # Check for keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('c') or key == ord(' '):  # 'c' or SPACE - CAPTURE
                print("\n" + "="*60)
                print("CAPTURING IMAGE...")
                print("="*60)
                captured_rgb = rgb.copy()
                is_captured = True
                
                # Process captured image
                print("Estimating depth...")
                captured_depth = depth_model.predict(captured_rgb)
                
                print("Creating 3D structure...")
                pcd = depth_to_pointcloud(captured_rgb, captured_depth, K)
                pcd = downsample_pcd(pcd, voxel_size=0.03)
                
                num_points = len(pcd.points)
                print(f"✓ Created 3D structure with {num_points} points")
                
                # Show detections if enabled
                if detector:
                    dets = detector.detect(captured_rgb)
                    if len(dets) > 0:
                        print(f"  Detected {len(dets)} objects")
                        captured_rgb_with_dets = draw_detections(captured_rgb.copy(), dets)
                    else:
                        captured_rgb_with_dets = captured_rgb.copy()
                else:
                    captured_rgb_with_dets = captured_rgb.copy()
                
                # Show captured image
                cv2.imshow('Webcam Input', cv2.cvtColor(captured_rgb_with_dets, cv2.COLOR_RGB2BGR))
                
                # Show depth map visualization
                depth_colored = depth_to_colormap(captured_depth)
                cv2.imshow('Depth Map', cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))
                
                # Update 3D structure
                if num_points > 0:
                    if not geometry_added:
                        vis.remove_geometry(pcd_geom, reset_bounding_box=False)
                        pcd_geom = pcd
                        vis.add_geometry(pcd_geom, reset_bounding_box=False)
                        geometry_added = True
                    else:
                        pcd_geom.points = pcd.points
                        pcd_geom.colors = pcd.colors
                        vis.update_geometry(pcd_geom)
                    
                    # Set view
                    view_ctl = vis.get_view_control()
                    center = pcd.get_center()
                    view_ctl.set_lookat(center)
                    view_ctl.set_front([0, 0, -1])
                    view_ctl.set_up([0, -1, 0])
                    bbox = pcd.get_axis_aligned_bounding_box()
                    extent = bbox.get_extent()
                    max_extent = max(extent)
                    if max_extent > 0:
                        zoom = 0.5 / max_extent
                        view_ctl.set_zoom(min(zoom, 0.9))
                    
                    bbox = pcd.get_axis_aligned_bounding_box()
                    center = pcd.get_center()
                    print(f"  Point cloud center: {center}")
                    print(f"  Bounding box: min={bbox.get_min_bound()}, max={bbox.get_max_bound()}")
                    print("="*60 + "\n")
                else:
                    print("WARNING: Point cloud is empty!")
                    print("="*60 + "\n")
                
                # Continue to show captured image for a moment
                continue
            
            elif key == ord('s'):  # Save current 3D structure
                if num_points > 0 and is_captured:
                    filename = f"3d_structure_{save_count:04d}.ply"
                    o3d.io.write_point_cloud(filename, pcd_geom)
                    print(f"✓ Saved 3D structure to {filename} ({num_points} points)")
                    save_count += 1
                else:
                    print("No captured image to save! Press 'c' to capture first.")
            
            # Only update 3D window if we have a captured image
            if is_captured and num_points > 0:
                vis.poll_events()
                vis.update_renderer()
            
            frame_count += 1
            elapsed = t.elapsed_ms()
            fps = 1000.0 / (elapsed + 1e-9)
            
            # Print status less frequently
            if frame_count % 30 == 0:
                if is_captured:
                    print(f"Live preview | Last capture: {num_points} points | Press 'c' for new capture")
                else:
                    print(f"Live preview ({fps:.1f} FPS) | Press 'c' to capture image")
            

    except KeyboardInterrupt:
        print('\nExiting...')
    finally:
        cam.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        print("Closed all windows. 3D structures saved if you exported any.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--intrinsics', type=str, default='../config/camera_intrinsics.yaml')
    parser.add_argument('--detect', action='store_true')
    parser.add_argument('--yolo_model', type=str, default='yolov8n.pt')
    args = parser.parse_args()
    # adjust default path to work when running from project root
    if args.intrinsics.startswith('..'):
        args.intrinsics = 'config/camera_intrinsics.yaml'
    main(args)
