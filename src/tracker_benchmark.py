"""Tracker Benchmark File

This script is used to benchmark tracker performance.
"""

import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

from acquisition import VideoStream
from utility_functions import crosshair
from detection import baseCorrection, houghDetect

from functools import partial
from itertools import combinations
from chrono import Timer

from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from motrackers.utils import draw_tracks

# Define each of the trackers
trackers = {}
trackers["sort"] = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
trackers["centroid"] = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')
trackers["centroidKF"] = CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge')
trackers["iou"] = IOUTracker(max_lost=3, min_detection_confidence=0.4, iou_threshold=0.5, tracker_output_format='mot_challenge')

# Define video stream
SCALE_FACTOR = 0.5
vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))

# Define detector
detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                                    param1 = 27, param2 = 19, 
                                    minRadius = 12, maxRadius = 15, debug = False)

# Start the video stream object
vs.start()

# Create empty lists for storing the data per tracker
trackers_list = [[],[],[],[]]
timers_list = [[],[],[],[]]

from copy import deepcopy
# Repeat per tracker (and restart video stream)
for idx, _tracker in enumerate(trackers.values()):

    # Create empty lists to store each attempt (for averaging)
    _trackers_list = [[],[],[]]
    _timers_list = [[],[],[]]

    # Create deep copies of the tracker to be used (so restart from 0)
    tracker = [deepcopy(_tracker), deepcopy(_tracker), deepcopy(_tracker)]

    for attempt_no in range(3):

        # Restart video stream properties each attempt
        vs.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_no = 0

        # Main logic loop
        while True:

            # Provide timing functionality for tracker loop
            with Timer() as full_timed:

                # Fetch next available frame from generator
                ret, frame = next(vs.read())

                # Check if valid return flag
                if not ret:
                    break
            
                # Detect vials
                ret, bboxes, points = detector_func(frame)

                # Extract bounding box coordinates
                detection_bboxes = np.asarray([b[:-1] for b in bboxes]).reshape((-1,5))
                bb_temp = []
                for b in detection_bboxes:
                    x1 = b[0]
                    x2 = b[2]
                    y1 = b[1]
                    y2 = b[3]
                    bb_temp.append([x1+ 0*(x2-x1)/2, y1 + 0*(y2-y1)/2, (x2-x1), y2-y1])

                # Update tracker (and draw bounding boxes)
                detection_bboxes = np.asarray(bb_temp, dtype="int")
                detection_confidences = np.ones(len(bboxes)).reshape((-1,))
                detection_class_ids = np.ones(len(bboxes)).reshape((-1,))

                # Time tracker update
                with Timer() as tracker_timed:
                    output_tracks = tracker[attempt_no].update(detection_bboxes, detection_confidences, detection_class_ids)

                frame = draw_tracks(frame.copy(), output_tracks)
                
                # Display frame contents
                cv2.imshow("Frame", frame)

                # Wait 16ms for user input - simulating 60FPS
                # Note - could synchronise with timer and moving average for more control
                key = cv2.waitKey(16) & 0xFF

                # Escape / close if "q" pressed
                if key == ord("q"):
                    break
            
            # Fetch the final tracker id to find total number of trackers instantiated
            tracker_ids = list(tracker[attempt_no].tracks.keys())
            final_id = tracker_ids[-1] if bool(tracker_ids) else 0

            # Append attempt data to list
            _trackers_list[attempt_no].append(int(final_id))
            _timers_list[attempt_no].append(tracker_timed.elapsed)
            
            # Temporarily exit from video stream early
            if frame_no == 200:
                break
            
            frame_no += 1

    # Assign mean values to lists
    trackers_list[idx] = np.mean(np.asarray(_trackers_list), axis=0).tolist()
    timers_list[idx] = np.mean(np.asarray(_timers_list), axis=0).tolist()

# Graph the tracker count for the different trackers
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Tracker Count")
ax.set_xlabel("Frame Number (-)")
t = np.arange(0, max([len(trk) for trk in trackers_list]))
ax.plot(t, trackers_list[0], label="SORT")
ax.plot(t, trackers_list[1], label="Centroid")
ax.plot(t, trackers_list[2], label="Centroid_KF")
ax.plot(t, trackers_list[3], label="IoU")
ax.legend()

# Graph the frame time for the different trackers
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Frame Time (ms)")

# Setting y limit to enalrge useful portion of chart
ax.set_ylim([0, 0.15])

ax.set_xlabel("Frame Number (-)")
t = np.arange(0, max([len(trk) for trk in timers_list]))
ax.plot(t, timers_list[0], label="SORT")
ax.plot(t, timers_list[1], label="Centroid")
ax.plot(t, timers_list[2], label="Centroid_KF")
ax.plot(t, timers_list[3], label="IoU")
ax.legend()

# Add annotation for 30FPS (realtime threshold)
import matplotlib.transforms as transforms
ax.axhline(y=0.0333, color='gray', linestyle='--')
trans = transforms.blended_transform_factory(
    ax.get_yticklabels()[0].get_transform(), ax.transData)
ax.text(0,0.0333, "30FPS", color="gray", transform=trans, 
        ha="right", va="center")

# Show plots
plt.show()

# Tidy up - close windows and stop video stream object
cv2.destroyAllWindows()
vs.stop()
