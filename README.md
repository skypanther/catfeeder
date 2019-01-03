# Cat feeder project

We've got four cats (yeah, we're crazy).

* Marmalade (ginger/orange tiger) has to eat on a regular schedule or she throws up.
* Penny (black &amp; white tuxedo) weighs in at 19 pounds, so she's on a diet of a controlled amount of weight-management type food.
* Bean (muted calico, mostly light gray) is our kitten, who we're trying to feed kitten food though she loves all the other cats' food more than hers.
* Chloe (dark gray tiger) is our only self-regulating eater.

It's a pain keeping up with their needs and schedule and trying to keep them from eating each other's food. This repo represents the rudimentary beginnings of an automatic cat feeder. 

My plan:

1. Detect when a cat is in view
2. Determine which cat is present
3. Open the appropriate door exposing the correct food bowl

I'll probably work in dispensing the correct amount of food daily too. Though, that's not a first-release goal.

Since our cats are each a unique and distinct color, I hope to differentiate based on HSV color ranges of the detected cat. 

## Current state

Very-Pre-Alpha &mdash; detects my cat and extracts HSV colors (with the plan to identify each cat by its color). The script is very slow and so far implements none of the IoT portions of the goals. 

### Requirements

* Python 3.6+
* OpenCV 3.4+ (4.x not tested)
* Numpy 1.14+
* imutils
* sklearn (SciKit-learn)

Current two methods in testing:

### Object detection with YOLOv3

Object detection with YOLO works and processes each video frame in about 2-3 seconds on my 2011 MacBook Pro (2.3 GHz i5, 8GB RAM, SSD).

Usage:

```python
python3 main.py
```

Uses the built-in webcam. Aim that camera towards a cat and it will output HSV color info and show the extracted region of the frame containing the cat.

**Note:** You'll need to download the yolov3.weights file and put it into the yolo-coco folder. Get it from https://pjreddie.com/media/files/yolov3.weights

### Object isolation with Mask R-CNN

Object isolation with Mask R-CNN works and processes each video frame in about 5-6 seconds on my 2011 MacBook Pro (2.3 GHz i5, 8GB RAM, SSD).

Usage:

```python
python3 main_rcnn.py
```

Uses the built-in webcam. Aim that camera towards a cat and it will output HSV color info and show the extracted region of the frame containing the cat.

The HSV color info output will include black as one of the dominant colors, which is not entirely accurate (unless your cat is black). The script isolates the non-rectangular region containing the cat, then fills the remaining portion of a bounding rectangle with black. You will need to take this into account if you use this technique.


----

**Note:** If you look through the history on this repo you'll find my former attempt, which used a custom Tensorflow model. That method worked, and was reasonably fast. However, it was not particularly accurate. And, it clearly showed how little I knew about Tensorflow at the time.

# License

MIT License (see the LICENSE file for full license statement)

&copy; 2018 Tim Poulsen

Some code from:

* https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/
* https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/