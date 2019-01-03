"""
Webcam library. Generally, don't access this library directly. Instead, use
the video.py library.
"""
import cv2
import threading
import time
from video.abstract_cam import AbstractCam


class Webcam(AbstractCam):
    def __init__(self, cam_id=0):
        self.cam_id = cam_id
        self.running = False
        self.current_frame = None

    def start(self):
        self.cam = cv2.VideoCapture(self.cam_id)
        self.ct = threading.Thread(target=self._capture_image_thread, name="Webcam")
        self.ct.daemon = True
        self.running = True
        self.ct.start()

    def stop(self):
        self.running = False
        self.ct.join()
        self.cam.release()

    def read_frame(self):
        if self.running is True:
            return self.current_frame

    def _capture_image_thread(self):
        while self.running is True:
            success, frame = self.cam.read()
            if success:
                self.current_frame = frame
            time.sleep(0.005)
