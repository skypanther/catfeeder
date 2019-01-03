"""
Cat isolater
Uses Mask R-CNN to isolate (perform image segmentation) on images to identify cats present in the image

Note: This turns out to be rather slow, like 5 seconds per frame on my (admittedly older) MacBook Pro.

Usage:

from cat_isolator import CatIsolater
cat_isolator = CatIsolater(model_path="mask-rcnn-coco", confidence=0.75, threshold=0.3)
image = cv2.imread("your_picture.jpg")
cats = cat_isolator.look_for_cat(image, debug=True)
if len(cats) > 0:
    print("Found {} cats".format(len(cats)))
    for cat in cats:
        # show the cats
        cv2.imshow("Extracted_Region", cat.extracted_region)
        cv2.waitKey(0)

"""
import cv2
import numpy as np
import os
import time


class CatIsolater:
    def __init__(self, model_path="coco", confidence=0.75, threshold=0.3):
        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = os.path.join(model_path, "frozen_inference_graph.pb")
        configPath = os.path.join(model_path, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        labelsPath = os.path.join(model_path, "object_detection_classes_coco.txt")
        self.net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
        self.confidence = confidence
        self.threshold = threshold
        self.labels = open(labelsPath).read().strip().split("\n")

    def look_for_cat(self, image, debug=False):
        H, W = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
        end = time.time()

        if debug:
            # print timing information and volume information on Mask R-CNN
            print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
            print("[INFO] boxes shape: {}".format(boxes.shape))
            print("[INFO] masks shape: {}".format(masks.shape))

        # loop over the number of detected objects
        cats = []
        for i in range(0, boxes.shape[2]):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]
            text = "{}: {:.4f}".format(self.labels[classID], confidence)
            if debug:
                print(text)

            if self.labels[classID] != "cat" or confidence < self.confidence:
                # we care only about cats, so skip anything else
                continue

            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            # extract the pixel-wise segmentation for the object, resize
            # the mask such that it's the same dimensions of the bounding
            # box, and then finally threshold to create a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_NEAREST)
            mask = (mask > self.threshold)
            roi = image[startY:endY, startX:endX]
            visMask = (mask * 255).astype("uint8")
            extracted_region = cv2.bitwise_and(roi, roi, mask=visMask)
            detection_response = DetectionResponse(text=text,
                                                   confidence=confidence,
                                                   extracted_region=extracted_region)
            cats.append(detection_response)
        return cats


class DetectionResponse:
    """
    Simple object representing a detected object
    """

    def __init__(self, text="", confidence=0.0, extracted_region=None):
        self.text = text
        self.confidence = confidence
        self.extracted_region = extracted_region
