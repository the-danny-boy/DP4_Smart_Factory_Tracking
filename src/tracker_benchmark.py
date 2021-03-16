"""Tracker Benchmark File

This script is used to benchmark tracker performance.
"""

from functools import partial
from chrono import Timer
from copy import deepcopy

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from acquisition import VideoStream
from utility_functions import crosshair
from YOLO_detector_wrapper import setup, detect_wrapper

from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from motrackers.utils import draw_tracks

# Benchmark settings
early_terminate = 200
repeat_attempts = 3

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
model = setup()
detector_func = partial(detect_wrapper, model=model, debug=False)

# Start the video stream object
vs.start()

# Create empty lists for storing the data per tracker
trackers_list = []
timers_list = []

for i in range(len(trackers)):
    trackers_list.append([])
    timers_list.append([])


# Repeat per tracker (and restart video stream)
for idx, _tracker in enumerate(trackers.values()):

    # Create empty lists to store each attempt (for averaging) and
    # Create deep copies of the tracker to be used (so restart from 0)
    _trackers_list = []
    _timers_list = []
    tracker = []
    for i in range(repeat_attempts):
        _trackers_list.append([])
        _timers_list.append([])
        tracker.append(deepcopy(_tracker))

    for attempt_no in range(repeat_attempts):

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
            
            # Break from loop early - for testing
            if bool(early_terminate):
                if frame_no > early_terminate:
                    break
            
            frame_no += 1

    # Assign mean values to lists
    trackers_list[idx] = np.mean(np.asarray(_trackers_list), axis=0).tolist()
    timers_list[idx] = np.mean(np.asarray(_timers_list) * 1000, axis=0).tolist()

# Graph the tracker count for the different trackers
labels = ["SORT", "Centroid", "Centroid_KF", "IoU"]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Tracker Count")
ax.set_xlabel("Frame Number (-)")
t = np.arange(0, max([len(trk) for trk in trackers_list]))

for i in range(len(labels)):
    ax.plot(t, trackers_list[i], label=labels[i])
ax.legend()
plt.tight_layout()

# Graph the frame time for the different trackers
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Frame Time (ms)")
ax.set_xlabel("Frame Number (-)")
t = np.arange(0, max([len(trk) for trk in timers_list]))

# Setting y limit to enalrge useful portion of chart
#ax.set_ylim([0, 0.15])

for i in range(len(labels)):
    ax.plot(t, timers_list[i], label=labels[i])
ax.legend()

# Show the 30FPS line (realtime threshold)
import matplotlib.transforms as transforms
ax.axhline(y=33.3, color='black', linestyle='--')
ax.annotate(text="30FPS", xy =(ax.get_xlim()[1], 33.3 - 
            ax.get_ylim()[1] * 0.01), xycoords="data", color="black")

# Enforce integer ticks on x-axis for discrete / integer frame numbers
import matplotlib.ticker as ticker
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.tight_layout()

# Show plots
plt.show()

# Tidy up - close windows and stop video stream object
cv2.destroyAllWindows()
vs.stop()
