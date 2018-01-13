# Cat feeder project


Rudimentary beginnings of an automatic cat feeder project. Goal:

1. Detect when a cat is in view
2. Determine if it's an orange cat
3. If so, open a door exposing her food bowl

Tensorflow model based on [Cat vs Dog detection](https://pythonprogramming.net/convolutional-neural-network-kats-vs-dogs-machine-learning-tutorial/)

Color detection from [OpenCV detect object by color range](https://thecodacus.com/opencv-object-tracking-colour-detection-python/)

Training data from:

* https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
* https://archive.ics.uci.edu/ml/datasets.html
* http://www.vision.caltech.edu/Image_Datasets/Caltech101/
* http://www.vision.caltech.edu/Image_Datasets/Caltech256/

Tensorboard:

`tensorboard --logdir=log --port 6006`