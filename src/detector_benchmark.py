"""Detector Benchmark File

This script is used to benchmark detector performance.
"""

import cv2
import numpy as np
import torch
import sys
sys.path.append('ScaledYOLOv4/')
from ScaledYOLOv4.utils.general import bbox_iou
import glob
from copy import deepcopy
from chrono import Timer

from detection import houghDetect
from functools import partial


# Paths for test images and labels
image_paths = r"ScaledYOLOv4/vials/test/images/"
label_paths = r"ScaledYOLOv4/vials/test/labels/"

# Find all items at target path
images = sorted(glob.glob(image_paths + "*.jpg"))
labels = sorted(glob.glob(label_paths + "*.txt"))

# Define detector(s)
detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                        param1 = 27, param2 = 19, 
                        minRadius = 12, maxRadius = 15, debug = False)

iou_thresh = 0.5

# List of dictionaries to encode detections
#Image = image index, Bbox = bbox coordinates, Conf = confidence
#TP/FP = binary flag indicating TP (1) or FP (0)
#detections = {"image":[], "bbox":[], "conf":[], "TP/FP":[]}
detections = []
gt_bboxes = []
total_positives = 0
for index, (img_path, label_path) in enumerate(zip(images, labels)):

    # Read image and resize to 50%
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    gt_bboxes.append([])
    # Read contents of GT label data
    with open(label_path) as file:
        bboxes = file.readlines()
        for bbox in bboxes:
            # Read data and split into separate variables
            obj_class, x, y, w, h = list(map(float, bbox.rstrip().split(" ")))

            # Undo normalisation of bbox
            x, w = int(x * img.shape[1]), int(w * img.shape[1])
            y, h = int(y * img.shape[0]), int(h * img.shape[0])

            # Convert bbox notation to corner points
            x1, x2 = int(x - w/2), int(x + w/2)
            y1, y2 = int(y - h/2), int(y + h/2)
            gt_bboxes[index].append((x1,y1,x2,y2))
    
    # Increment total number of GT detections
    total_positives += len(gt_bboxes[index])

    # Perform detection
    with Timer() as timed:
        ret, bboxes, points = detector_func(img)
    detection_time = timed.elapsed
    #print("Time elapsed is:", detection_time)
    
    for bbox in bboxes:
        x1, y1, x2, y2, conf, class_id = bbox
        _image = index
        _bbox = (x1,y1,x2,y2)
        _conf = conf

        # Iterate through all gt_bboxes, calculating IoU to identify if TP/FP
        flag = True
        for gt_bbox in gt_bboxes[index]:
            iou = bbox_iou(torch.FloatTensor(bbox[:4]), torch.FloatTensor(gt_bbox))

            if iou > iou_thresh:
                detections.append({"image":_image, "bbox":_bbox, "conf":_conf, "TP/FP":1})
                flag = False
                break

        if flag:
            detections.append({"image":_image, "bbox":_bbox, "conf":_conf, "TP/FP":0})

# Then sort the detections by decreasing confidence
sorted_detections = sorted(detections, key=lambda x: x["conf"], reverse=True)

# Find the accumulated TPs, FPs
# Iterate through all detections in order of decreasing confidence
for index, det in enumerate(sorted_detections):

    # If first entry, assign accumulators as current TP/FP status
    if index == 0:
        det["Acc_TP"] = 1 if det["TP/FP"] else 0
        det["Acc_FP"] = 0 if det["TP/FP"] else 1
    
    # Otherwise, find current TP/FP status, and add to previous accumulator states
    else:
        delta_accTP = 1 if det["TP/FP"] else 0
        delta_accFP = 0 if det["TP/FP"] else 1
        det["Acc_TP"] = sorted_detections[index-1]["Acc_TP"] + delta_accTP
        det["Acc_FP"] = sorted_detections[index-1]["Acc_FP"] + delta_accFP

    # Calculate the precision (TP/(TP+FP)) and recall (TP/(TP+FN))
    TP, FP = det["Acc_TP"], det["Acc_FP"]
    precision = TP / (TP + FP)
    recall = TP / total_positives
    det["Precision"] = precision
    det["Recall"] = recall
    #f1 = (precision * recall) / ((precision + recall) / 2) if (precision + recall != 0) else np.nan

"""
# Plot precision-recall curve
import matplotlib.pyplot as plt
plt.plot([d["Recall"] for d in sorted_detections],  
         [d["Precision"] for d in sorted_detections])
plt.title("Precision x Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()
"""

# Find the 11-point interpolated AP (average precision)
# = 1/11(SUM(precision@recall_[0.0,0.1,0.2,...,0.8,0.9,1.0]))
