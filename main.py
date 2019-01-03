import cv2
import imutils
import numpy as np
from cat_detector import CatDetector
from color_analyzer import ColorAnalyzer
from video_stream import VideoStream

cat_detector = CatDetector(model_path="yolo-coco", confidence=0.5, threshold=0.3)

# camera = cv2.VideoCapture(0)
camera = VideoStream(source='webcam', cam_id=0)
# ipcam = VideoStream(source='ipcam', ipcam_url='http://10.15.18.101/mjpg/video.mjpg')
# picam = VideoStream(source='picam')
# jetson = VideoStream(source='jetson', resolution=(1920, 1080))

camera.start()

cv2.namedWindow("Frame")
cv2.moveWindow("Frame", 40, 320)
cv2.namedWindow("Extracted_Region")
cv2.moveWindow("Extracted_Region", 380, 40)
# cv2.namedWindow("Masked")
# cv2.moveWindow("Masked", 380, 320)

MIN_ORANGE_AREA = 300  # min area of orange to assume Marmalade present
color_analyzer = ColorAnalyzer(MIN_ORANGE_AREA)
# set range of colors to detect
# from get_dominant_colors, we find:
# HSV colors:
# [(145, 87, 0),
#  (129, 108, 45),
#  (120, 81, 110),
#  (116, 71, 182),
#  (121, 25, 243)]

lowerBound = np.array([115, 50, 120])  # dark orange
upperBound = np.array([135, 255, 255])  # bright orange
color_analyzer.set_color_range(lowerBound, upperBound)

is_first_run = True
while True:
    # frame = camera.read_frame()
    image = camera.read()
    cv2.imshow("Frame", imutils.resize(image, width=480))
    if is_first_run:
        cv2.waitKey(2000)
        is_first_run = False

    # do stuff with the frame
    cats = cat_detector.look_for_cat(image, debug=True)
    if len(cats) > 0:
        print("Found {} cats".format(len(cats)))
        for cat in cats:
            # show the cats
            if cat.extracted_region is not None:
                h, w = cat.extracted_region.shape[:2]
                if h > 0 and w > 0:
                    cv2.imshow("Extracted_Region", cat.extracted_region)
            if color_analyzer.is_right_cat_visible(cat.extracted_region):
                print("I found a big orange cat! Et tu Marmalade?")
            rgb, hsv = color_analyzer.get_dominant_colors(cat.extracted_region)
            print("HSV colors:")
            print(hsv)
            cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        camera.stop()
        break
