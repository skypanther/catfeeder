import cv2
import numpy as np

lowerBound = np.array([3, 30, 120])  # 13, 100%, 42%
upperBound = np.array([53, 255, 255])  # 53, 100%, 42%

cv2.namedWindow("maskClose")
cv2.moveWindow("maskClose", 40, 320)
cv2.namedWindow("mask")
cv2.moveWindow("mask", 380, 40)
cv2.namedWindow("maskOpen")
cv2.moveWindow("maskOpen", 380, 320)

cam = cv2.VideoCapture(0)
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # hsv = input('Lower Bound HSV: ')
    # if hsv is not "":
    #     lowerBound = np.array([int(x) for x in hsv.split(',')])
    ret, img = cam.read()
    if img is None:
        print('no image')
        cv2.waitKey(10)
        continue
    img = cv2.resize(img, (340, 220))

    # convert BGR to HSV
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create the Mask
    mask = cv2.inRange(imgHSV, lowerBound, upperBound)
    # morphology
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

    maskFinal = maskClose
    _, conts, h = cv2.findContours(
        maskFinal.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img, conts, -1, (255, 0, 0), 3)

    if len(conts):
        for i in range(len(conts)):
            if cv2.contourArea(conts[i]) > 1800:
                print("big orange area, probably Marmalade")

    # for i in range(len(conts)):
    #     x, y, w, h = cv2.boundingRect(conts[i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.putText(img,  str(i + 1), (x, y + h), font, 50, 255)
    cv2.imshow("maskClose", maskClose)
    cv2.imshow("maskOpen", maskOpen)
    cv2.imshow("mask", mask)
    cv2.imshow("cam", img)
    cv2.waitKey(100)
