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

from utility_functions import crosshair, heron, regularity
from itertools import combinations

import scipy.spatial

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
        #crosshair(frame, object["positions"][-1], size = 8, color = (0,0,255))
        pass


        """ Insert metrics code here ================================ """



        """ ========================================================= """


    # VOIDS (SPATIAL DESCRIPTOR)
    with Timer() as void_timed:
        
        # Check if enough (3) points for triangulation
        if len(points) > 3:

            # Calculate delaunay triangulation
            ts = scipy.spatial.Delaunay(points)

            # Select traingles
            points = np.asarray(points)
            pts_shaped = points[ts.simplices]

            # Combine heron (area) and regularity functions, and map to point list
            comb_funcs = lambda x: (heron(x[0], x[1], x[2]), regularity(x[0], x[1], x[2]))
            comb_outputs = list(map(comb_funcs, pts_shaped))
            ar, reg = map(list, zip(*comb_outputs)) 

            # Filter bad traingle points using logical numpy operator
            bad_pts = pts_shaped[ np.logical_and((np.asarray(ar)>800), (np.asarray(reg) > 0.1)) ]

            # Visualise this as a 50% opacity overlay
            _frame = frame.copy()
            for bp in bad_pts:
                cv2.fillPoly(_frame, [bp], (0,0,255))
            frame = cv2.addWeighted(frame, 0.5, _frame, 0.5, 0)

    print("Void process time (ms):", void_timed.elapsed)

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
