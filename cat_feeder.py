import cv2
import numpy as np
import os
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# const and variable definitions
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'more_data_5layer'
LOOP_DELAY = 500  # 1.5 seconds
MIN_ORANGE_AREA = 1200  # min area of orange to assume Marmalade present
cam = cv2.VideoCapture(0)  # ref to our web cam
kernelOpen = np.ones((5, 5))  # for drawing the "open" mask
kernelClose = np.ones((20, 20))  # for draing the "closed" mask
lowerBound = np.array([3, 30, 120])  # dark orange
upperBound = np.array([53, 255, 255])  # bright orange

cv2.namedWindow("maskClose")
cv2.moveWindow("maskClose", 40, 320)
cv2.namedWindow("mask")
cv2.moveWindow("mask", 380, 40)
cv2.namedWindow("maskOpen")
cv2.moveWindow("maskOpen", 380, 320)

if not os.path.exists('{}.meta'.format(MODEL_NAME)):
    print("Missing model, quitting...")
    exit()

# set up our model and convnet
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(
    convnet,
    optimizer='adam',
    learning_rate=LR,
    loss='categorical_crossentropy',
    name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load(MODEL_NAME)

# Main loop, logic is:
# Capture image, run it through our TF model
# If a cat is detected, see if it's orange
# If so, open access to the food bowl
# Else, repeat the loop after a short delay
while True:
    ret, img = cam.read()
    if img is None:
        print('no image')
        cv2.waitKey(LOOP_DELAY)
        continue
    resized_image = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    resized_image_grayscale = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    data = resized_image_grayscale.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        print('I see a cat!')
        # we have detected a cat
        # convert BGR to HSV
        imgHSV = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
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

        cv2.drawContours(resized_image, conts, -1, (255, 0, 0), 3)
        cv2.imshow("maskClose", maskClose)
        cv2.imshow("maskOpen", maskOpen)
        cv2.imshow("mask", mask)
        cv2.imshow("cam", resized_image)

        if len(conts):
            for i in range(len(conts)):
                if cv2.contourArea(conts[i]) > MIN_ORANGE_AREA:
                    print("big orange area, probably Marmalade")

    cv2.waitKey(LOOP_DELAY)
