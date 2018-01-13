"""
Attempted model improvement with additions from
Tensorflow docs - failed miserably :-)
Accuracy was roughly 0%
"""
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = 'train_two'
TEST_DIR = 'test_two'
IMG_SIZE = 60
LR = 1e-3
MODEL_NAME = 'even_moar_data'
TRAIN_DATA_NAME = 'train_data_{}.npy'.format(MODEL_NAME)
TEST_DATA_NAME = 'test_data_{}.npy'.format(MODEL_NAME)


def create_label(image_name):
    """ Create an one-hot encoded vector from image name """
    word_label = image_name.split('.')[-3]
    if word_label == 'cat':
        return np.array([1, 0])
    else:
        return np.array([0, 1])


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        if img == ".DS_Store":
            continue
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            continue
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save(TRAIN_DATA_NAME, training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            continue
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), img_num])
    shuffle(testing_data)
    np.save(TEST_DATA_NAME, testing_data)
    return testing_data


if os.path.exists(TRAIN_DATA_NAME):
    train_data = np.load(TRAIN_DATA_NAME)
else:
    train_data = create_train_data()
if os.path.exists(TEST_DATA_NAME):
    test_data = np.load(TEST_DATA_NAME)
else:
    test_data = create_test_data()

train = train_data[:-500]
test = train_data[-500:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(
    convnet,
    32,
    5,
    padding="same",
    activation=tf.nn.relu
)
pool1 = tf.layers.max_pooling2d(inputs=convnet, pool_size=[2, 2], strides=2)

convnet2 = conv_2d(
    pool1,
    64,
    5,
    padding="same",
    activation=tf.nn.relu
)
pool2 = tf.layers.max_pooling2d(inputs=convnet2, pool_size=[2, 2], strides=2)

convnet3 = conv_2d(
    pool2,
    1024,
    5,
    padding="same",
    activation=tf.nn.relu
)

# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)

# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)

# convnet = conv_2d(convnet, 1024, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)

convnet_f = fully_connected(convnet3, 2, activation='softmax')
convnet_f = regression(
    convnet_f,
    optimizer='adam',
    learning_rate=LR,
    loss='categorical_crossentropy',
    name='targets')

model = tflearn.DNN(convnet_f, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
else:
    train = train_data[:-500]
    test = train_data[-500:]
    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train]
    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in test]
    model.fit(
        {'input': X},
        {'targets': Y},
        n_epoch=10,
        validation_set=({'input': test_x}, {'targets': test_y}),
        snapshot_step=500,
        show_metric=True,
        run_id=MODEL_NAME)
    model.save(MODEL_NAME)

# test the model vs the actual test data
fig = plt.figure(figsize=(16, 12))
shuffle(test_data)
for num, data in enumerate(test_data[:16]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(4, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]

    print('File: {}.jpg - [{}]'.format(img_num, model_out))

    if np.argmax(model_out) == 1:
        str_label = 'Not cat'
    else:
        str_label = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()
