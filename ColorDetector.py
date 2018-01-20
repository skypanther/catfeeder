import cv2
import numpy as np


class ColorDetector:
    IMG_WIDTH = 640
    IMG_HEIGHT = 480
    kernelOpen = np.ones((5, 5))  # for drawing the "open" mask
    kernelClose = np.ones((20, 20))  # for draing the "closed" mask

    def __init__(self, min_area):
        self.MIN_ORANGE_AREA = min_area

    def set_color_range(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound

    def setup_preview_windows(self):
        cv2.namedWindow("maskClose")
        cv2.moveWindow("maskClose", 40, self.IMG_HEIGHT)
        cv2.namedWindow("mask")
        cv2.moveWindow("mask", self.IMG_WIDTH, 40)
        cv2.namedWindow("maskOpen")
        cv2.moveWindow("maskOpen", self.IMG_WIDTH, self.IMG_HEIGHT)

    def is_right_cat_visible(self, img, show_windows=False):
        resized_image = cv2.resize(img, (self.IMG_WIDTH, self.IMG_WIDTH))
        imgHSV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        # create the mask
        mask = cv2.inRange(
            imgHSV,
            self.lowerBound,
            self.upperBound)
        # remove noise with morphological "open"
        maskOpen = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernelOpen)
        # close up internal holes in contours with "close"
        maskClose = cv2.morphologyEx(
            maskOpen,
            cv2.MORPH_CLOSE,
            self.kernelClose)

        _, conts, h = cv2.findContours(
            maskClose.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        if show_windows:
            cv2.drawContours(resized_image, conts, -1, (255, 0, 0), 3)
            cv2.imshow("maskClose", maskClose)
            cv2.imshow("maskOpen", maskOpen)
            cv2.imshow("mask", mask)
            cv2.imshow("cam", resized_image)

        if len(conts):
            for i in range(len(conts)):
                if cv2.contourArea(conts[i]) > self.MIN_ORANGE_AREA:
                    return True
        return False
