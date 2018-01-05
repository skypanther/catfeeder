import cv2
import numpy as np
from ObjectDetector import ObjectDetector
from ColorDetector import ColorDetector

# const and variable definitions
MODEL_NAME = 'more_data_5layer'
LOOP_DELAY = 500  # 1.5 seconds
LONG_WAIT = 1000 * 60 * 3  # 3 mins
MIN_ORANGE_AREA = 1200  # min area of orange to assume Marmalade present
CAM = cv2.VideoCapture(0)  # ref to our web cam
SHOW_PREVIEW_WINDOWS = True

object_detector = ObjectDetector(MODEL_NAME)
color_detector = ColorDetector(MIN_ORANGE_AREA)

# set range of colors to detect
lowerBound = np.array([3, 30, 120])  # dark orange
upperBound = np.array([53, 255, 255])  # bright orange
color_detector.setColorRange(lowerBound, upperBound)

if SHOW_PREVIEW_WINDOWS:
    color_detector.setupPreviewWindows()

# Main loop, logic is:
# Capture image, run it through our TF model
# If a cat is detected, see if it's orange
# If so, open access to the food bowl
# Else, repeat the loop after a short delay
while True:
    loop_wait = LOOP_DELAY
    ret, img = CAM.read()
    if img is None:
        print('no image')
        cv2.waitKey(LOOP_DELAY)
        continue

    if object_detector.isCatVisible(img):
        # we have detected a cat
        print('I see a cat!')

        if color_detector.isRightCatVisible(img, SHOW_PREVIEW_WINDOWS):
            print("big orange area, probably Marmalade")
            # open food bowl
            # set loop_wait to a higher time so bowl doesn't
            # shut too quickly
            # loop_wait = LONG_WAIT

    cv2.waitKey(loop_wait)
