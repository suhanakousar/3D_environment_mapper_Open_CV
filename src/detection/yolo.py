# Simple wrapper for YOLOv8 using ultralytics
from ultralytics import YOLO
import numpy as np


class YOLOv8Detector:
    def __init__(self, model='yolov8n.pt', device=''):
        # device: '' -> auto, 'cpu' or '0' for cuda:0
        self.model = YOLO(model)

    def detect(self, rgb_uint8):
        # rgb_uint8: HxWx3
        results = self.model(rgb_uint8, imgsz=640, conf=0.35)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                detections.append({'bbox': xyxy, 'conf': conf, 'class': cls})
        return detections
