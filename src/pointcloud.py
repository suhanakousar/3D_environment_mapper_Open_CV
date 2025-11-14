import numpy as np
import open3d as o3d


def depth_to_pointcloud(rgb, depth, K):
    """
    rgb: HxWx3 uint8 (RGB)
    depth: HxW float (meters)
    K: 3x3 intrinsic matrix
    returns: Open3D PointCloud
    """
    h, w = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.reshape(-1)
    x = (i.reshape(-1) - cx) * z / fx
    y = (j.reshape(-1) - cy) * z / fy

    pts = np.stack((x, y, z), axis=1)
    # filter out invalids (z <= 0)
    valid = z > 0.01
    pts = pts[valid]

    colors = rgb.reshape(-1, 3)[valid].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def downsample_pcd(pcd, voxel_size=0.02):
    return pcd.voxel_down_sample(voxel_size)
