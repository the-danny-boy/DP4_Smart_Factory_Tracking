"""Detection Tuning Utility

This script provides a GUI for tuning detector parameters
"""

import numpy as np
import cv2
import glob

from acquisition import VideoStream
from detection import houghDetect, hsvDetect, templateMatch
from utility_functions import crosshair


#Empty callback function
def nothing(x):
    pass


# Create required trackbars, with specified parameter ranges
def trackbarSetup(window_name, detector_type):

    if detector_type == "hough":
        parameters = {"dp":[10,20], #Note: scaled by 10, as integer trackbars
                      "minDist":[0,30],
                      "param1":[0,300],
                      "param2":[0,20],
                      "minRadius":[0,30],
                      "maxRadius":[0,30]}

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
        parameters = {"threshold":[80,100], #Note: scaled by 100
                      "template_img [static, average_crop]": [0,1]} 

    for p, v in zip(parameters.keys(), parameters.values()):
        cv2.createTrackbar(p, window_name, v[0], v[1], nothing)

    return parameters


if __name__ == "__main__":
    SCALE_FACTOR = 0.5

    # Boolean flag for whether using video footage (else - training images)
    vid = False

    # Define video stream
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))

    # Grab all image paths
    image_paths = r"ScaledYOLOv4/vials/train/images/"
    images = sorted(glob.glob(image_paths + "*.jpg"))
    
    # The selector index for which detector to use
    detector_sel = 1

    # Define available detectors (and reference the functions)
    detector_types = ["hough", "hsv", "template"]
    detector_funcs = [houghDetect, hsvDetect, templateMatch]

    # Assertion to check valid selector used
    assert detector_sel < len(detector_types), "DETECTOR SELECTOR INVALID (OUT OF RANGE)"

    # Select the appropriate detector (if available)
    detector_type = detector_types[detector_sel]
    detector_func = detector_funcs[detector_sel]
    
    # Create a window with the allocated name
    window_name = "Detector Tuning Utility : " + detector_type
    cv2.namedWindow(window_name)    

    # Create trackbar to scrape through frames
    if vid:
        vs.start()
        cv2.createTrackbar("Frame No.", window_name, 0, int(vs.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), nothing)
    else:
        cv2.createTrackbar("Frame No.", window_name, 0, len(images)-1, nothing)

    parameters = trackbarSetup(window_name, detector_type)
    parameter_settings = [None] * len(parameters.keys())

    # Main logic loop
    while True:

        # Fetch target frame using trackbar
        frame_no=cv2.getTrackbarPos("Frame No.", window_name)


        # Check if need to acquire video or image, and handle accordingly
        if vid:
            vs.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = next(vs.read())
            if not ret:
                break
        else:
            frame = cv2.imread(images[frame_no])
            frame = cv2.resize(frame, (int(frame.shape[1] * SCALE_FACTOR), int(frame.shape[0] * SCALE_FACTOR)))
    

        # Fetch detector parameters from the trackbars
        for index, parameter in enumerate(parameters.keys()):
            parameter_settings[index] = cv2.getTrackbarPos(parameter, window_name)

        # Apply detector (and show debug visualisation window)
        ret, bboxes, points = detector_func(frame, *parameter_settings, debug = True)
        
        # If successful detection, visualise with blue crosshairs in same window
        if ret:
            for point in points:
                crosshair(frame, point, color = (255,0,0))

        # Display frame contents
        cv2.imshow(window_name, frame)

        # Escape / close if "q" pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # Tidy up - close windows and stop video stream object
    cv2.destroyAllWindows()
