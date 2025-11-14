"""
Enhanced visualization utilities.
"""
import numpy as np
import cv2
import open3d as o3d


def depth_to_colormap(depth: np.ndarray, min_depth: float = None, max_depth: float = None) -> np.ndarray:
    """
    Convert depth map to colorized visualization.
    
    Args:
        depth: Depth map (H, W)
        min_depth: Minimum depth for normalization (None = auto)
        max_depth: Maximum depth for normalization (None = auto)
        
    Returns:
        Colorized depth map (H, W, 3) uint8
    """
    if min_depth is None:
        min_depth = depth.min()
    if max_depth is None:
        max_depth = depth.max()
    
    # Normalize to 0-255
    depth_norm = np.clip((depth - min_depth) / (max_depth - min_depth + 1e-8), 0, 1)
    depth_norm = (depth_norm * 255).astype(np.uint8)
    
    # Apply colormap (JET for depth visualization)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
    
    return depth_colored


def draw_detections(rgb: np.ndarray, detections: list, class_names: dict = None) -> np.ndarray:
    """
    Draw detection bounding boxes on RGB image.
    
    Args:
        rgb: RGB image (H, W, 3) uint8
        detections: List of detection dicts with 'bbox', 'conf', 'class'
        class_names: Optional dict mapping class IDs to names
        
    Returns:
        RGB image with drawn detections
    """
    if class_names is None:
        # Default YOLO class names (COCO dataset)
        class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            # Add more as needed
        }
    
    img = rgb.copy()
    
    for det in detections:
        bbox = det['bbox'].astype(int)
        conf = det['conf']
        cls = det['class']
        
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_names.get(cls, f'class_{cls}')}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


def create_multi_view(rgb: np.ndarray, depth: np.ndarray, detections: list = None, 
                     width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a multi-view display showing RGB, depth, and detections.
    
    Args:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth map (H, W)
        detections: Optional list of detections
        width: Output width
        height: Output height
        
    Returns:
        Combined image (H, W, 3) uint8
    """
    # Resize inputs to match
    rgb_resized = cv2.resize(rgb, (width, height))
    
    # Create depth visualization
    depth_colored = depth_to_colormap(depth)
    depth_resized = cv2.resize(depth_colored, (width, height))
    
    # Draw detections on RGB if provided
    if detections:
        rgb_with_dets = draw_detections(rgb_resized, detections)
    else:
        rgb_with_dets = rgb_resized
    
    # Combine horizontally: RGB | Depth
    combined = np.hstack([rgb_with_dets, depth_resized])
    
    return combined

