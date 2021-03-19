"""Detection Tuning Utility

This script provides a GUI for tuning metric (void) parameters
"""

import numpy as np
import cv2
from acquisition import VideoStream
from utility_functions import crosshair
from YOLO_detector_wrapper import setup, detect_wrapper
from utility_functions import crosshair, heron, regularity
from itertools import combinations
import scipy.spatial
from functools import partial

#Empty callback function
def nothing(x):
    pass

if __name__ == "__main__":
    SCALE_FACTOR = 0.5
    #vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-03-18_20h40m_Camera1_011.webm", 
                     fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))
        
    # Define detector
    model = setup()
    detector_func = partial(detect_wrapper, model=model, debug=False)

    # Start the video stream object
    vs.start()
    
    # Create a window with the allocated name
    window_name = "Metric Tuning Utility"
    cv2.namedWindow(window_name)    

    # Create trackbar to scrape through frames
    cv2.createTrackbar("Frame No.", window_name, 0, int(vs.cap.get(cv2.CAP_PROP_FRAME_COUNT)-1), nothing)
    cv2.createTrackbar("Ok Regularity", window_name, 0, 100, nothing)
    cv2.createTrackbar("Bad Regularity", window_name, 0, 100, nothing)
    cv2.createTrackbar("Ok Area", window_name, 0, 1000, nothing)
    cv2.createTrackbar("Ok Area_Reg", window_name, 0, 1000, nothing)
    cv2.createTrackbar("Bad Area", window_name, 0, 1000, nothing)
    cv2.createTrackbar("Bad Area_Reg", window_name, 0, 1000, nothing)

    # Main logic loop
    while True:

        # Fetch target frame using trackbar
        frame_no=cv2.getTrackbarPos("Frame No.", window_name)
        vs.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = next(vs.read())

        # Check if valid return flag
        if not ret:
            break
    
        # Apply detector (and show debug visualisation window)
        ret, bboxes, points = detector_func(frame)
        _frame = frame.copy()        

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

            # Colorise good packing as green (colour all using convex hull)
            conv = points[scipy.spatial.ConvexHull(points).vertices]
            cv2.fillPoly(_frame, [conv], (0,255,0))

            # Define criteria for good / ok / bad packing
            # Frame 683
            """
            area_ok = 400
            reg_ok = 0.05
            area_bad = 800
            reg_bad = 0.1
            """
            area_ok = cv2.getTrackbarPos("Ok Area", window_name) # 896 -> 553
            area_ok_reg = cv2.getTrackbarPos("Ok Area_Reg", window_name) # 552 -> 153
            reg_ok = cv2.getTrackbarPos("Ok Regularity", window_name) / 100 # 5 => 0.005 -> 0.1
            area_bad_reg = cv2.getTrackbarPos("Bad Area", window_name) # 666 -> 858
            area_bad = cv2.getTrackbarPos("Bad Area_Reg", window_name) # 699 -> 625
            reg_bad = cv2.getTrackbarPos("Bad Regularity", window_name) / 100 # 10 => 0.01 -> 0.46

            # Filter other severities of packing using logical numpy operator
            ok_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_ok), np.logical_and((np.asarray(ar)>area_ok_reg), (np.asarray(reg) > reg_ok))) ]
            bad_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_bad), np.logical_and((np.asarray(ar)>area_bad_reg), (np.asarray(reg) > reg_bad))) ]

            for op in ok_pts:
                cv2.fillPoly(_frame, [op], (0,255,255))

            for bp in bad_pts:
                cv2.fillPoly(_frame, [bp], (0,0,255))

            # Ideally sample barycentric coordinates for smooth transitions (but slow)
            _frame = cv2.GaussianBlur(_frame, (15,15), 0)

            # Visualise this as a 50% opacity overlay
            frame = cv2.addWeighted(frame, 0.5, _frame, 0.5, 0)

        for p in points:
            crosshair(frame, p, size = 8, color = (0,0,0))

        # Display frame contents
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        # Escape / close if "q" pressed
        if key == ord("q"):
            break
    
    # Tidy up - close windows and stop video stream object
    cv2.destroyAllWindows()
    vs.stop()