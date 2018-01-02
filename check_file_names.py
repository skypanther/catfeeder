import os
import cv2

TRAIN_DIR = 'train_two'
TEST_DIR = 'test_two'


def check_directory(dir='train_two', num_parts=3):
    for img in os.listdir(dir):
        path = os.path.join(dir, img)
        if len(img.split('.')) != num_parts:
            print('{}'.format(img))
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            print('{} - bad image'.format(img))


print('Checking training directory:')
check_directory(TRAIN_DIR, 3)

print('Checking testing directory:')
check_directory(TEST_DIR, 2)
