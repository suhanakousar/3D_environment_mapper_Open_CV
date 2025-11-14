# midas.py
# Lightweight wrapper to load MiDaS using torch.hub or the official repo.

import torch
import cv2
import numpy as np


class MiDaSDepth:
    def __init__(self, model_type='DPT_Large', device=None):
        # model_type options: 'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Use torch.hub to load MiDaS (will download weights on first run)
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if model_type == 'MiDaS_small':
            self.transform = self.transforms.small_transform
        else:
            self.transform = self.transforms.default_transform

    def predict(self, rgb_uint8: np.ndarray):
        # rgb_uint8: HxWx3 uint8 RGB
        img = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
        inp = self.transform(img).to(self.device)
        with torch.no_grad():
            prediction = self.model(inp)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth = prediction.cpu().numpy()
        # Post-process: normalize to positive range, scale to meters with arbitrary scale (user may calibrate)
        depth = np.maximum(depth, 1e-6)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        # scale to a default range (0.2m - 5m) — this is arbitrary and for visualization
        depth_m = 0.2 + depth * (5.0 - 0.2)
        return depth_m
