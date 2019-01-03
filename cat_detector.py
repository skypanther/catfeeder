"""
Cat detector
Uses YOLOv2 for object detection on images to identify cats present in the image

Usage:

from cat_detector import CatDetector
cat_detector = CatDetector(model_path="mask-rcnn-coco", confidence=0.75, threshold=0.3)
image = cv2.imread("your_picture.jpg")
image = cv2.resize(frame, (480, 480))  # resize for better performance
cats = cat_detector.look_for_cat(image, debug=True)
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


class CatDetector:
    def __init__(self, model_path="yolo-coco", confidence=0.75, threshold=0.3):
        # derive the paths to the Mask R-CNN weights and model configuration
        weightsPath = os.path.join(model_path, "yolov3.weights")
        configPath = os.path.join(model_path, "yolov3.cfg")
        labelsPath = os.path.join(model_path, "coco.names")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.confidence = confidence
        self.threshold = threshold
        self.labels = open(labelsPath).read().strip().split("\n")

    def look_for_cat(self, image, debug=False):
        H, W = image.shape[:2]
        # The blob size (360x360) has to match that set in yolov3.cfg
        # or you'll get an error / crash like:
        # Inconsistent shape for ConcatLayer in function 'getMemoryShapes'
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (480, 480),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()
        if debug:
            # print timing information and volume information on Mask R-CNN
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # loop over the number of detected objects
        cats = []

        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if self.labels[classID] != "cat" or confidence < self.confidence:
                    # we care only about cats, so skip anything else
                    continue

                if debug:
                    print(">> I'm {}% sure I see a {}".format(round(confidence * 100, 2), self.labels[classID]))

                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                text = "{}: {:.4f}".format(self.labels[classID], confidence)
                # extract the bounding box coordinates
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                startX = x
                startY = y

                extracted_region = np.zeros((h, w, 3), dtype=np.uint8)
                extracted_region = image[startY:startY + h, startX:startX + w]

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
