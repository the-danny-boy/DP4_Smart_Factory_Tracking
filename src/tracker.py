import math

import numpy as np
import cv2

from acquisition import VideoStream
from utility_functions import crosshair
from detection import baseCorrection, houghDetect

from functools import partial
from itertools import combinations
from chrono import Timer

from motrackers import CentroidTracker, IOUTracker, CentroidKF_Tracker, SORT
from motrackers.utils import draw_tracks


# Define tracker
#tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.3)
#tracker = CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge')
tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')

if __name__ == "__main__":
    SCALE_FACTOR = 0.5
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                     fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))
    
    detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                                     param1 = 27, param2 = 19, 
                                     minRadius = 12, maxRadius = 15, debug = False)

    # Start the video stream object
    vs.start()

    # Main logic loop
    while True:

        # Provide timing functionality for tracker loop
        with Timer() as timed:

            # Fetch next available frame from generator
            ret, frame = next(vs.read())

            # Check if valid return flag
            if not ret:
                break
        
            # Detect vials
            ret, bboxes, points = detector_func(frame)

            # Extract bounding box coordinates
            detection_bboxes = np.asarray([b[:-1] for b in bboxes]).reshape((-1,4))
            bb_temp = []
            for b in detection_bboxes:
                x1 = b[0]
                x2 = b[2]
                y1 = b[1]
                y2 = b[3]
                bb_temp.append([x1+ 0*(x2-x1)/2, y1 + 0*(y2-y1)/2, (x2-x1), y2-y1])

            # Draw bounding boxes
            detection_bboxes = np.asarray(bb_temp, dtype="int")
            detection_confidences = np.ones(len(bboxes)).reshape((-1,))
            detection_class_ids = np.ones(len(bboxes)).reshape((-1,))
            output_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
            frame = draw_tracks(frame.copy(), output_tracks)

            
            # Display frame contents
            cv2.imshow("Frame", frame)

            # Wait 16ms for user input - simulating 60FPS
            # Note - could synchronise with timer and moving average for more control
            key = cv2.waitKey(16) & 0xFF

            # Escape / close if "q" pressed
            if key == ord("q"):
                break
    
        print("Elapsed Time:", timed.elapsed)

    # Tidy up - close windows and stop video stream object
    cv2.destroyAllWindows()
    vs.stop()