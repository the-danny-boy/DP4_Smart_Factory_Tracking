"""Combined Benchmark File

This script is used to benchmark overall performance of the system.
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

#_tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')
_tracker = Centroid_Tracker(max_lost=3)

# Start the video stream object
vs.start()

# Create empty (nested) lists for storing the data per run
acquisition_times = []
detection_times = []
tracker_times = []
total_times = []
tracker = []

for i in range(repeat_attempts):
    acquisition_times.append([])
    detection_times.append([])
    tracker_times.append([])
    total_times.append([])

    # Create deep copies of the tracker to be used (so restart from 0)
    tracker.append(deepcopy(_tracker))


for attempt_no in range(repeat_attempts):

    # Restart video stream properties each attempt
    vs.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_no = 0
    objectID = 1

    # Main logic loop
    while True:

        # Provide overall timer for full frame processing 
        with Timer() as full_timed:

            # Time the acquisition step
            with Timer() as acquisition_timed:
                ret, frame = next(vs.read())
                if not ret:
                    break

            # Time the detection
            with Timer() as detector_timed:
                ret, bboxes, points = detector_func(frame)

            # Time tracker update
            with Timer() as tracker_timed:
                objectID, trackedObjects = tracker[attempt_no].update(objectID, points)

            """
            # Iterate through tracked objects and annotate
            for idx, object in zip(trackedObjects.keys(), trackedObjects.values()):
                crosshair(frame, object["positions"][-1], size = 8, color = (0,0,255))

            # Show annotated frame
            cv2.imshow("Frame", frame)
            
            # Only wait for 1ms to limit the performance overhead
            key = cv2.waitKey(1) & 0xFF

            # Escape / close if "q" pressed
            if key == ord("q"):
                break
            """

            
            # Sleep operation to add overhead
            if bool(fake_overhead):
                time.sleep(fake_overhead)
            

        # Print elapsed time
        print("Total time elapsed (ms): ", full_timed.elapsed*1000)

        # Append attempt data to list
        acquisition_times[attempt_no].append(acquisition_timed.elapsed)
        detection_times[attempt_no].append(detector_timed.elapsed)
        tracker_times[attempt_no].append(tracker_timed.elapsed)
        total_times[attempt_no].append(full_timed.elapsed)

        
        # Break from loop early - for testing
        if bool(early_terminate):
            if frame_no > early_terminate:
                break
        
        frame_no += 1

# Assign mean values to lists
acquisition_times_avg = np.mean(np.asarray(acquisition_times) * 1000, axis=0).tolist()
detection_times_avg = np.mean(np.asarray(detection_times) * 1000, axis=0).tolist()
tracker_times_avg = np.mean(np.asarray(tracker_times) * 1000, axis=0).tolist()
total_times_avg = np.mean(np.asarray(total_times) * 1000, axis=0).tolist()

# Graph the frame time contribution
import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.set_ylabel("Frame Time (ms)")
ax.set_xlabel("Frame Number (-)")

ax.set_ylim(0,80)
ax.set_xlim(0,len(total_times_avg)-1)

# Generate array for x data
t = np.arange(0, len(total_times_avg))

# Show a stacked plot for the other variables on top
ax.stackplot(t, acquisition_times_avg, detection_times_avg, tracker_times_avg, 
            alpha=1.0, labels=["Acquisition", "Detection", "Tracking"])

# Shade the area under the total frame (purple, lighten using alpha)
ax.fill_between(t, total_times_avg, 0, facecolor="purple", 
            color="purple", lw=0, alpha=0.7, label = "Overhead", zorder=-1)

# Display the legend - consider placing top left for extended run?
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
