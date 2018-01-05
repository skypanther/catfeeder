import cv2
import numpy as np
import os
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class ObjectDetector:
    IMG_SIZE = 50
    LR = 1e-3

    def __init__(self, name):
        self.MODEL_NAME = name
        if not os.path.exists('{}.meta'.format(self.MODEL_NAME)):
            print("Missing model, quitting...")
            raise 'MissingModel'

        # set up our model and convnet
        tf.reset_default_graph()
        convnet = input_data(
            shape=[None, self.IMG_SIZE, self.IMG_SIZE, 1],
            name='input')
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
            learning_rate=self.LR,
            loss='categorical_crossentropy',
            name='targets')

        self.MODEL = tflearn.DNN(convnet, tensorboard_dir='log')
        self.MODEL.load(self.MODEL_NAME)

    def isCatVisible(self, img):
        resized_image = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
        resized_image_grayscale = cv2.cvtColor(
            resized_image,
            cv2.COLOR_BGR2GRAY)

        data = resized_image_grayscale.reshape(self.IMG_SIZE, self.IMG_SIZE, 1)
        model_out = self.MODEL.predict([data])[0]

        if np.argmax(model_out) == 0 and model_out[0] > 0.9:
            # print(model_out)
            # print('I see a cat!')
            return True
        return False
