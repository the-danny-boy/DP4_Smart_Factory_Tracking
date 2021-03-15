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

# Dictionary to encode detections
#Image = image index, Bbox = bbox coordinates, Conf = confidence
#TP/FP = binary flag indicating TP (1) or FP (0)
detections = {"image":[], "bbox":[], "conf":[], "TP/FP":[]}

for index, (img_path, label_path) in enumerate(zip(images, labels)):

    # Read image and resize to 50%
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

    gt_bboxes = []
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
            gt_bboxes.append((x1,y1,x2,y2))
    
    # Perform detection
    with Timer() as timed:
        ret, bboxes, points = detector_func(img)
    detection_time = timed.elapsed
    #print("Time elapsed is:", detection_time)
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, conf, class_id = bbox
        detections["image"].append(index)
        detections["bbox"].append((x1,y1,x2,y2))
        detections["conf"].append(conf)

        # Iterate through all gt_bboxes, calculating IoU to identify if TP/FP
        flag = True
        for ip, gt_bbox in enumerate(gt_bboxes):
            iou = bbox_iou(torch.FloatTensor(bbox[:4]), torch.FloatTensor(gt_bbox))

            if iou > iou_thresh:
                detections["TP/FP"].append(1)
                flag = False
                break

        if flag:
            detections["TP/FP"].append(0)
    

# Then sort the detections by decreasing confidence
# Find the accumulated TPs, FPs
# Calculate the precision and recall

# Find the 11-point interpolated AP (average precision)
# = 1/11(SUM(precision@recall_[0.0,0.1,0.2,...,0.8,0.9,1.0]))
