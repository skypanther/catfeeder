"""
Raspberry Pi camera library. Generally, don't access this library directly.
Instead, use the video.py library.
"""
import threading
from video.abstract_cam import AbstractCam
from picamera.array import PiRGBArray
from picamera import PiCamera


class Picam(AbstractCam):
    def __init__(self, resolution=(320, 240)):
        self.resolution = resolution
        self.running = False
        self.current_frame = None

    def start(self):
        self.cam = PiCamera()
        self.rawCapture = PiRGBArray(self.cam, size=self.resolution)
        self.stream = self.cam.capture_continuous(self.rawCapture,
                                                  format="bgr",
                                                  use_video_port=True)
        self.ct = threading.Thread(target=self._capture_image_thread,
                                   name="Picam")
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
        for f in self.stream:
            self.current_frame = f.array
            self.rawCapture.truncate(0)
