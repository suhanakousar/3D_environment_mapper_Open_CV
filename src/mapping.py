"""
Point cloud accumulation and map management.
"""
import numpy as np
import open3d as o3d
from typing import Optional


class PointCloudMap:
    """Accumulates point clouds over time to build a global map."""
    
    def __init__(self, voxel_size=0.02, max_points=1000000):
        """
        Args:
            voxel_size: Voxel size for downsampling (meters)
            max_points: Maximum number of points before aggressive downsampling
        """
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.global_map = o3d.geometry.PointCloud()
        self.frame_count = 0
        
    def add_frame(self, pcd: o3d.geometry.PointCloud, max_age_frames: Optional[int] = None):
        """
        Add a new point cloud frame to the global map.
        
        Args:
            pcd: Point cloud to add
            max_age_frames: If set, only keep points from last N frames
        """
        # Downsample the new frame
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # Add to global map
        self.global_map += pcd_down
        
        # Remove outliers (statistical)
        if len(self.global_map.points) > 100:
            self.global_map, _ = self.global_map.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
        
        # Aggressive downsampling if too many points
        if len(self.global_map.points) > self.max_points:
            self.global_map = self.global_map.voxel_down_sample(self.voxel_size * 1.5)
            print(f"Downsampled global map to {len(self.global_map.points)} points")
        
        self.frame_count += 1
        
    def get_map(self) -> o3d.geometry.PointCloud:
        """Get the current global map."""
        return self.global_map
    
    def clear(self):
        """Clear the global map."""
        self.global_map.clear()
        self.frame_count = 0
        
    def save(self, filename: str):
        """Save the global map to a file."""
        o3d.io.write_point_cloud(filename, self.global_map)
        print(f"Saved map with {len(self.global_map.points)} points to {filename}")
        
    def load(self, filename: str):
        """Load a map from a file."""
        self.global_map = o3d.io.read_point_cloud(filename)
        print(f"Loaded map with {len(self.global_map.points)} points from {filename}")
        
    def get_stats(self) -> dict:
        """Get statistics about the current map."""
        if len(self.global_map.points) == 0:
            return {"points": 0, "frames": self.frame_count}
        
        bbox = self.global_map.get_axis_aligned_bounding_box()
        return {
            "points": len(self.global_map.points),
            "frames": self.frame_count,
            "bbox_min": bbox.get_min_bound().tolist(),
            "bbox_max": bbox.get_max_bound().tolist(),
            "center": self.global_map.get_center().tolist()
        }

