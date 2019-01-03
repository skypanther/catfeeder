"""
Jetson gstreamer camera library. Generally, don't access this library directly.
Instead, use the video.py library.
"""
import cv2
import threading
import time
from video.abstract_cam import AbstractCam


class Jetsoncam(AbstractCam):
    def __init__(self, resolution=(1920, 1080)):
        self.running = False
        self.current_frame = None
        w, h = resolution
        self.cameraString = ('nvcamerasrc ! '
                             'video/x-raw(memory:NVMM), '
                             'width=(int)2592, height=(int)1458, '
                             'format=(string)I420, framerate=(fraction)30/1 ! '
                             'nvvidconv ! '
                             'video/x-raw, width=(int){}, height=(int){}, '
                             'format=(string)BGRx ! '
                             'videoconvert ! appsink').format(w, h)

    def start(self):
        self.cam = cv2.VideoCapture(self.cameraString, cv2.CAP_GSTREAMER)
        self.ct = threading.Thread(target=self._capture_image_thread,
                                   name="Jetsoncam")
        self.ct.daemon = True
        self.running = True
        self.ct.start()

    def stop(self):
        self.running = False
        self.ct.join()
        self.stream.close()
        self.rawCapture.close()
        self.cam.close()

    def read_frame(self):
        if self.running is True:
            return self.current_frame

    def _capture_image_thread(self):
        while self.running is True:
            success, frame = self.cam.read()
            if success:
                self.current_frame = frame
            time.sleep(0.005)
