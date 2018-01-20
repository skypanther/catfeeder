import cv2
import numpy as np
from ObjectDetector import ObjectDetector
from ColorDetector import ColorDetector
# from FeederController import FeederController

# const and variable definitions
MODEL_NAME = 'models/even_moar_data'
STARTING_LOOP_DELAY = 1500  # 1.5 seconds
LONG_WAIT = 1000 * 60 * 3  # 3 mins
"""
min area of orange to assume Marmalade present
this is in sq px within a 480w x 320h image
"""
MIN_ORANGE_AREA = 1200
CAM = cv2.VideoCapture(0)  # ref to our web cam
SHOW_PREVIEW_WINDOWS = True

# GPIO pins
# FEEDER_OPEN_PIN = 3
# FEEDER_CLOSE_PIN = 4
# FEEDER_PIR_PIN = 5
# WAIT_FOR_CAT_EATING_DELAY = 30

object_detector = ObjectDetector(MODEL_NAME)
color_detector = ColorDetector(MIN_ORANGE_AREA)
# fd = FeederController(FEEDER_OPEN_PIN, FEEDER_CLOSE_PIN, FEEDER_PIR_PIN)

# set range of colors to detect
lowerBound = np.array([3, 30, 120])  # dark orange
upperBound = np.array([53, 255, 255])  # bright orange
color_detector.set_color_range(lowerBound, upperBound)

loop_delay = STARTING_LOOP_DELAY

if SHOW_PREVIEW_WINDOWS:
    color_detector.setup_preview_windows()


def reset_loop_delay():
    loop_delay = STARTING_LOOP_DELAY


# Main loop, logic is:
# Capture image, run it through our TF model
# If a cat is detected, see if it's orange
# If so, open access to the food bowl
# Else, repeat the loop after a short delay
while True:
    ret, img = CAM.read()
    if img is None:
        # no image captured, wait then try again
        cv2.waitKey(loop_delay)
        continue

    if object_detector.is_cat_visible(img):
        # we have detected a cat
        print('I see a cat!')

        if color_detector.is_right_cat_visible(img, SHOW_PREVIEW_WINDOWS):
            print("big orange area, probably Marmalade")
            """
            # TODO: need to pause the while loop here for that 30 sec delay
            # then resume it once the cat is no longer present
            # the following won't work
            fd.open_food_bowl()
            loop_delay = 0
            fd.watch_cat_presence(WAIT_FOR_CAT_EATING_DELAY, reset_loop_delay)
            """

    cv2.waitKey(loop_delay)
