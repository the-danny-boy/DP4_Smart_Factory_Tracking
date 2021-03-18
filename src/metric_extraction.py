"""Metric Extraction File

This script is used to extract metrics from the video.
"""

import numpy as np
import cv2
from functools import partial
from chrono import Timer
import time
from copy import deepcopy

from acquisition import VideoStream
from utility_functions import crosshair
from YOLO_detector_wrapper import setup, detect_wrapper
from centroid_tracker import Centroid_Tracker

# Benchmark settings
early_terminate = 200
repeat_attempts = 3
fake_overhead = 0.02

# Define video stream
SCALE_FACTOR = 0.5
vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-03-18_20h40m_Camera1_011.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))

# Define detector
model = setup()
detector_func = partial(detect_wrapper, model=model, debug=False)

tracker = Centroid_Tracker(max_lost=3)

# Start the video stream object
vs.start()

frame_no = 0
objectID = 1

# Main logic loop
while True:

    # Acquire next frame
    ret, frame = next(vs.read())
    if not ret:
        break

    # Detect Objects in Frame
    ret, bboxes, points = detector_func(frame)

    # Update Tracker
    objectID, trackedObjects = tracker.update(objectID, points)

    # Iterate through tracked objects and annotate
    for idx, object in zip(trackedObjects.keys(), trackedObjects.values()):
        crosshair(frame, object["positions"][-1], size = 8, color = (0,0,255))


        """ Insert metrics code here ================================ """



        """ ========================================================= """



    # Show annotated frame
    cv2.imshow("Frame", frame)
    
    # Only wait for 1ms to limit the performance overhead
    key = cv2.waitKey(1) & 0xFF

    # Escape / close if "q" pressed
    if key == ord("q"):
        break
    
    frame_no += 1

# Tidy up - close windows and stop video stream object
cv2.destroyAllWindows()
vs.stop()
