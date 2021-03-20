"""Main File (Entrypoint)

This script serves as an entrypoint to the technology demo.
"""

import numpy as np
import cv2

from functools import partial

from acquisition import VideoStream
from detection import houghDetect, hsvDetect, templateMatch
from process import StreamProcessor

# Create a video stream object with video file and specified frame rate and frame size
SCALE_FACTOR = 0.5
vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))

detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                                     param1 = 27, param2 = 19, 
                                     minRadius = 12, maxRadius = 15, debug = False)

sp = StreamProcessor(src = vs, detector = detector_func)

# Main logic loop
while True:

    # Process next frame
    ret, frame = sp.update(analyse = False)

    # Check if valid return flag
    if not ret:
        break

    # Display frame contents
    cv2.imshow("Frame", frame)

    # Wait 16ms for user input - simulating 60FPS
    # Note - could synchronise with timer and moving average for more control
    key = cv2.waitKey(16) & 0xFF

    # Escape / close if "q" pressed
    if key == ord("q"):
        break

# Tidy up - close windows and stop stream processor object
cv2.destroyAllWindows()
sp.stop()
