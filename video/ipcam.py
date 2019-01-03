"""
IP camera library. Generally, don't access this library directly. Instead, use
the video.py library.
"""
import cv2
import threading
import time
from video.abstract_cam import AbstractCam


class IPcam(AbstractCam):
    def __init__(self, ipcam_url=''):
        self.ipcam_url = ipcam_url
        self.running = False
        self.current_frame = None

    def start(self):
        self.cam = cv2.VideoCapture(self.ipcam_url)
        self.ct = threading.Thread(target=self._capture_image_thread, name="IPcam")
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
