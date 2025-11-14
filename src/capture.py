import cv2


class CameraCapture:
    def __init__(self, cam_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError('Failed to read frame from camera')
        # OpenCV returns BGR — convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb

    def release(self):
        self.cap.release()
