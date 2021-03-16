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

from detection import houghDetect, templateMatch, hsvDetect
from functools import partial

# Paths for test images and labels
image_paths = r"ScaledYOLOv4/vials/test/images/"
label_paths = r"ScaledYOLOv4/vials/test/labels/"

# Find all items at target path
images = sorted(glob.glob(image_paths + "*.jpg"))
labels = sorted(glob.glob(label_paths + "*.txt"))

# Define detector(s)
hough = partial(houghDetect, dp = 1.5, minDist = 20, 
                        param1 = 27, param2 = 19, 
                        minRadius = 12, maxRadius = 15, debug = False)

hsv = partial(hsvDetect, hue_low = 0, hue_high = 179, 
                     sat_low = 0, sat_high = 94, 
                     val_low = 56, val_high = 255, debug = False)

template0 = partial(templateMatch, match_threshold = 30, template_path_idx = 0)
template1 = partial(templateMatch, match_threshold = 60, template_path_idx = 1)

# Initialise lists for detectors, AP scores, and detection times
detector_funcs = [hough, hsv, template0, template1]
detector_aps = []
detection_times = []

# Set the IoU threshold for TP detection
iou_thresh = 0.5

# Iterate through detection functions
for detector_func in detector_funcs:

    # Initialise variables
    gt_bboxes = []
    total_positives = 0
    _detection_times = []

    # "detections" is list of dictionaries to encode detections
    #Image = image index, Bbox = bbox coordinates, Conf = confidence
    #TP/FP = binary flag indicating TP (1) or FP (0)
    #detections = [{"image":[]}, {"bbox":[]}, {"conf":[]}, {"TP/FP":[]}]
    detections = []

    # Iterate through all images and labels for detections
    for index, (img_path, label_path) in enumerate(zip(images, labels)):
        # print(index)
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
        
        # Increment total number of GT detections => this is TP + FN
        total_positives += len(gt_bboxes[index])

        # Perform detection, and log detection time
        with Timer() as timed:
            ret, bboxes, points = detector_func(img)
        _detection_times.append(timed.elapsed)
        
        # Iterate through frames and store information in "detections" list
        for bbox in bboxes:
            x1, y1, x2, y2, conf, class_id = bbox
            _image = index
            _bbox = (x1,y1,x2,y2)
            _conf = conf

            # Iterate through all gt_bboxes, calculating IoU to identify if TP/FP
            flag = True
            for gt_bbox in gt_bboxes[index]:
                iou = bbox_iou(torch.FloatTensor(bbox[:4]), torch.FloatTensor(gt_bbox))

                # If IoU exceeds threshold, then true positive
                if iou > iou_thresh:
                    detections.append({"image":_image, "bbox":_bbox, "conf":_conf, "TP/FP":1})
                    flag = False
                    break
            
            # Flag remaining active indicated false positive
            if flag:
                detections.append({"image":_image, "bbox":_bbox, "conf":_conf, "TP/FP":0})

    # Then sort the detections by decreasing confidence
    sorted_detections = sorted(detections, key=lambda x: x["conf"], reverse=True)

    # Iterate through detections to find the accumulated metrics
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

    # Find the Average Precision (AP)
    ap = np.trapz([d["Precision"] for d in sorted_detections], \
        [d["Recall"] for d in sorted_detections])
    
    # Append detector metrics to list
    detection_times.append(np.mean(_detection_times))
    detector_aps.append(ap)

# Print out metrics
print("Detectors: Hough, HSV, Template (Raw), Template (Averaged)")
print("Detection Time(s):", detection_times)
print("Detection AP(s):", detector_aps)