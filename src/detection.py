"""Detection File

This script provides a collection of detection functions.
"""

import numpy as np
import cv2

from acquisition import VideoStream

# Hough Circle Detection Function
def houghDetect(frame):
    pass

# Grayscale Detection Function
def grayscaleDetect(frame):
    pass

# HSV Detection Function
def hsvDetect(frame, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
    try:

        # Blur the frame to reduce noise
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        # Convert the colour space of the input frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Convert input parameters into threshold hsv arrays
        hsv_low = np.array([hue_low, sat_low, val_low])
        hsv_high = np.array([hue_high, sat_high, val_high])

        # Mask the frame based on the hsv thresholds
        mask = cv2.inRange(hsv, hsv_low, hsv_high)

        # Segment the frame using the binary mask
        hsv_seg = cv2.bitwise_and(frame, frame, mask = mask)
        return True, hsv_seg
    
    except:
        return None, False

# Template Match Detection
def templateMatch(frame):
    pass

# Perform a sweep to benchmark detectors
def benchmark_detector(frame, detector):
    ret, detections = detector()
    pass


if __name__ == "__main__":
    SCALE_FACTOR = 0.5
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                     fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))
    
    # Start the video stream object
    vs.start()

    # Main logic loop
    while True:

        # Fetch next available frame from generator
        ret, frame = next(vs.read())

        # Check if valid return flag
        if not ret:
            break
    
        # Detect vials

        # Display frame contents
        cv2.imshow("Frame", frame)

        # Wait 16ms for user input - simulating 60FPS
        # Note - could synchronise with timer and moving average for more control
        key = cv2.waitKey(16) & 0xFF

        # Escape / close if "q" pressed
        if key == ord("q"):
            break
    
    # Tidy up - close windows and stop video stream object
    cv2.destroyAllWindows()
    vs.stop()