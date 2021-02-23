"""Detection Tuning Utility

This script provides a GUI for tuning detector parameters
"""

import numpy as np
import cv2

from acquisition import VideoStream
from detection import houghDetect, grayscaleDetect, hsvDetect, templateMatch


#Empty callback function
def nothing(x):
    pass


# Create required trackbars, with specified parameter ranges
def trackbarSetup(window_name, detector_type):

    if detector_type == "hough":
        pass
    elif detector_type == "gray":
        pass

    elif detector_type == "hsv":
        parameters = {"Hue Low":[0, 179],
                  "Hue High":[0, 179],
                  "Saturation Low":[0,255],
                  "Saturation High":[0, 255],
                  "Value Low":[0, 255],
                  "Value High":[0,255]}

    elif detector_type == "template":
        pass

    for p, v in zip(parameters.keys(), parameters.values()):
        cv2.createTrackbar(p, window_name, v[0], v[1], nothing)

    return parameters


if __name__ == "__main__":
    SCALE_FACTOR = 0.5
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                     fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))
    detector_sel = 2 # The selector index for which detector to use

    # Start the video stream object
    vs.start()

    # Define available detectors (and reference the functions)
    detector_types = ["hough", "gray", "hsv", "template"]
    detector_funcs = [houghDetect, grayscaleDetect, hsvDetect, templateMatch]

    # Assertion to check valid selector used
    assert detector_sel < len(detector_types), "DETECTOR SELECTOR INVALID (OUT OF RANGE)"

    # Select the appropriate detector (if available)
    detector_type = detector_types[detector_sel]
    detector_func = detector_funcs[detector_sel]

    # Create a window with the allocated name
    window_name = "Detector Tuning Utility : " + detector_type
    cv2.namedWindow(window_name)    

    # Create trackbar to scrape through frames
    cv2.createTrackbar("Frame No.", window_name, 0, int(vs.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), nothing)
    parameters = trackbarSetup(window_name, detector_type)
    parameter_settings = [None] * len(parameters.keys())

    # Main logic loop
    while True:

        # Fetch next available frame from generator

        frame_no=cv2.getTrackbarPos("Frame No.", window_name)
        vs.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

        ret, frame = next(vs.read())

        # Check if valid return flag
        if not ret:
            break
    
        # Fetch detector parameters from the trackbars
        for index, p in enumerate(parameters.keys()):
            parameter_settings[index] = cv2.getTrackbarPos(p, window_name)

        # Apply detection
        ret, frame = detector_func(frame, *parameter_settings)

        # If detection fails, proceed to next frame
        if not ret:
            continue

        # Display frame contents
        cv2.imshow(window_name, frame)

        # Wait 16ms for user input - simulating 60FPS
        # Note - could synchronise with timer and moving average for more control
        key = cv2.waitKey(16) & 0xFF

        # Escape / close if "q" pressed
        if key == ord("q"):
            break
    
    # Tidy up - close windows and stop video stream object
    cv2.destroyAllWindows()
    vs.stop()