"""
Multi-threaded video frame acquisition - one thread continuously grabs frames
while the other pulls individual frames from a queue

Examples:

from robovision import robovision as rv

vs_webcam = rv.VideoStream(cam_id=0)
vs_ipcam = rv.VideoStream(source='ipcam', ipcam_url=''http://10.15.18.101/mjpg/video.mjpg')
vs_picam = rv.VideoStream(source='picam')
vs_jetson = rv.VideoStream(source='jetson', resolution=(1920, 1080))

# use any of the above in this way to continuously read frames from the camera
vs_webcam.start()
while some_condition is True:
    frame = vs_webcam.read()
    # do stuff with the frame
vs_webcam.stop()

"""
from video.webcam import Webcam
# from video.picam import Picam
from video.ipcam import IPcam
# from video.jetsoncam import Jetsoncam

valid_sources = 'webcam', 'ipcam', 'picam', 'jetson'


class VideoStream:
    def __init__(self, source='webcam', cam_id=0, resolution=(320, 240), ipcam_url=''):
        if source not in valid_sources:
            print('Valid sources are {}'.format(', '.join(valid_sources)))
            raise 'InvalidSource'
        if source == 'ipcam':
            self.source = IPcam(ipcam_url=ipcam_url)
        # elif source == 'picam':
        #     self.source = Picam(resolution=resolution)
        # elif source == 'jetson':
        #     self.source = Jetsoncam()
        else:
            self.source = Webcam(cam_id=cam_id)

    def start(self):
        self.source.start()

    def stop(self):
        self.source.stop()

    def read_frame(self):
        return self.source.read_frame()

    def read(self):
        return self.source.read_frame()
