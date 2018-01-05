import cv2
import numpy as np


class ColorDetector:
    IMG_SIZE = 320
    kernelOpen = np.ones((5, 5))  # for drawing the "open" mask
    kernelClose = np.ones((20, 20))  # for draing the "closed" mask
    # lowerBound = np.array([3, 30, 120])  # dark orange
    # upperBound = np.array([53, 255, 255])  # bright orange

    def __init__(self, min_area):
        self.MIN_ORANGE_AREA = min_area

    def setColorRange(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound

    def setupPreviewWindows(self):
        cv2.namedWindow("maskClose")
        cv2.moveWindow("maskClose", 40, 320)
        cv2.namedWindow("mask")
        cv2.moveWindow("mask", 380, 40)
        cv2.namedWindow("maskOpen")
        cv2.moveWindow("maskOpen", 380, 320)

    def isRightCatVisible(self, img, showWindows=False):
        resized_image = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        imgHSV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        # create the Mask
        mask = cv2.inRange(
            imgHSV,
            self.lowerBound,
            self.upperBound)
        # morphology
        maskOpen = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernelOpen)
        maskClose = cv2.morphologyEx(
            maskOpen,
            cv2.MORPH_CLOSE,
            self.kernelClose)

        maskFinal = maskClose
        _, conts, h = cv2.findContours(
            maskFinal.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)

        if showWindows:
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
