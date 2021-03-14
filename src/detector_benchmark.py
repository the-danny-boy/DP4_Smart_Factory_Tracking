"""Detector Benchmark File

This script is used to benchmark detector performance.
"""

import cv2
import numpy

from detection import houghDetect
from functools import partial

# Paths for test images and labels (temporarily hardcoded)
img_path = "ScaledYOLOv4/vials/test/images/0.jpg"
label_path = "ScaledYOLOv4/vials/test/labels/0.txt"

# Read image and resize to 50%
img = cv2.imread(img_path)
img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)))

# Define detector(s)
detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                                    param1 = 27, param2 = 19, 
                                    minRadius = 12, maxRadius = 15, debug = False)


# Perform detection
ret, bboxes, points = detector_func(img)
for bbox in bboxes:
    
    x1, y1, x2, y2, conf, class_id = bbox

    # Print coordinate values
    # print(x1, y1, x2, y2)

    # Draw rectangle in green for detection visualisation
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)


# Read contents of GT label data
with open(label_path) as file:
    bboxes = file.readlines()


# Iterate through detections in label file
for bbox in bboxes:

    # Read data and split into separate variables
    obj_class, x, y, w, h = list(map(float, bbox.rstrip().split(" ")))

    # Undo normalisation of bbox
    x = int(x * img.shape[1])
    w = int(w * img.shape[1])
    y = int(y * img.shape[0])
    h = int(h * img.shape[0])

    # Convert bbox notation to corner points
    x1 = int(x - w/2)
    x2 = int(x + w/2)
    y1 = int(y - h/2)
    y2 = int(y + h/2)

    # Print coordinate values
    # print(x1, y1, x2, y2)

    # Draw rectangle in red for GT visualisation
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)


# Show image until keypress
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

