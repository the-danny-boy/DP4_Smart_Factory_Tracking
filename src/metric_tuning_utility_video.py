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

from copy import deepcopy
import math

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
        blank_canvas = np.zeros(_frame.shape)
        orig_points = deepcopy(points)     

        # Check if enough (3) points for triangulation
        if len(points) > 3:

            ppp = deepcopy(points)

            contour = np.asarray([[0,212], [260,487], [404, 486], [400, 327], [410, 328], [959, 147], [959, 538], [0, 537]])
            contour = np.array(contour).reshape((-1,1,2)).astype(np.int32)
            raw_pts = np.asarray([[0,212], [260,487], [404, 486], [400, 327], [410, 328], [959, 147], [959, 538], [0, 537], [0,212]])


            allPt = []
            subdiv_distance = 30

            refPt = deepcopy(raw_pts)

            point1 = np.asarray(refPt)
            point2 = np.roll(point1, -1, axis = 0)

            for j in range(len(point1)-1):
                
                p1 = point1[j]
                p2 = point2[j]
                
                _p1 = list(map(int, p1))
                allPt.append(_p1)

                dist = np.linalg.norm(p2 - p1)
                
                if dist > subdiv_distance:
                    dir_vector = np.subtract(p2, p1)
                    sample_no = math.ceil(dist / subdiv_distance) # makes sure one extra => dense end
                    fragd = subdiv_distance / dist

                    for i in range(sample_no):
                        vecPt = p1 + np.multiply(fragd * i, dir_vector)
                        _vecPt = list(map(int, vecPt))
                        allPt.append(_vecPt)

                else:

                    _p2 = list(map(int, p2))
                    allPt.append(_p2)


            points.extend(allPt)


            # Calculate delaunay triangulation
            ts = scipy.spatial.Delaunay(points)

            # Select triangles
            points = np.asarray(points)
            pts_shaped = points[ts.simplices]


            px = np.asarray([(ptt[0][0] + ptt[1][0] + ptt[2][0]) / 3 for ptt in pts_shaped], dtype="int").reshape(-1,1)
            py = np.asarray([(ptt[0][1] + ptt[1][1] + ptt[2][1]) / 3 for ptt in pts_shaped], dtype="int").reshape(-1,1)

            pts = np.concatenate((px,py),axis = 1)


            in_contour = [cv2.pointPolygonTest(contour, tuple(pttt), False) for pttt in pts]
            pts = pts[np.asarray(in_contour) < 0]
            pts_shaped = pts_shaped[np.asarray(in_contour) < 0]

            # Combine heron (area) and regularity functions, and map to point list
            comb_funcs = lambda x: (heron(x[0], x[1], x[2]), regularity(x[0], x[1], x[2]))
            comb_outputs = list(map(comb_funcs, pts_shaped))
            ar, reg = map(list, zip(*comb_outputs)) 

            # Colorise good packing as green (colour all using convex hull)
            conv = points[scipy.spatial.ConvexHull(points).vertices]
            cv2.fillPoly(_frame, [conv], (0,255,0))
            cv2.fillPoly(blank_canvas, [conv], (0,255,0))

            cv2.drawContours(blank_canvas,[contour],0,(0,0,0),-1)

            # Define criteria for good / ok / bad packing
            # Frame 683
            area_ok = cv2.getTrackbarPos("Ok Area", window_name) # 896 -> 553
            area_ok_reg = cv2.getTrackbarPos("Ok Area_Reg", window_name) # 552 -> 153
            reg_ok = cv2.getTrackbarPos("Ok Regularity", window_name) / 100 # 5 => 0.005 -> 0.1
            area_bad_reg = cv2.getTrackbarPos("Bad Area", window_name) # 666 -> 858
            area_bad = cv2.getTrackbarPos("Bad Area_Reg", window_name) # 699 -> 625
            reg_bad = cv2.getTrackbarPos("Bad Regularity", window_name) / 100 # 10 => 0.01 -> 0.46

            # Filter other severities of packing using logical numpy operator
            warn_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_ok), np.logical_and((np.asarray(ar)>area_ok_reg), (np.asarray(reg) > reg_ok))) ]
            crit_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_bad), np.logical_and((np.asarray(ar)>area_bad_reg), (np.asarray(reg) > reg_bad))) ]

            for wp in warn_pts:
                cv2.fillPoly(_frame, [wp], (0,255,255))
                cv2.fillPoly(blank_canvas, [wp], (255,0,0))

            for cp in crit_pts:
                cv2.fillPoly(_frame, [cp], (0,0,255))
                cv2.fillPoly(blank_canvas, [cp], (0,0,255))

            # Ideally sample barycentric coordinates for smooth transitions (but slow)
            _frame = cv2.GaussianBlur(_frame, (15,15), 0)


            # Combine adjacent regions and count
            good_canvas = cv2.inRange(blank_canvas, (0,10,0), (0,255,0))
            good_canvas = cv2.erode(good_canvas, (5,5))
            _good_count, _labels = cv2.connectedComponents(good_canvas)
            good_count = _good_count - 1
                
            crit_canvas = cv2.inRange(blank_canvas, (0,0,10), (0,0,255))
            crit_canvas = cv2.erode(crit_canvas, (5,5))
            _crit_count, _labels = cv2.connectedComponents(crit_canvas)
            crit_count = _crit_count - 1
                
            warn_canvas = cv2.inRange(blank_canvas, (10,0,0), (255,0,0))
            warn_canvas = cv2.erode(warn_canvas, (5,5))
            _warn_count, _labels = cv2.connectedComponents(warn_canvas)
            warn_count = _warn_count - 1


            # Visualise this as a 50% opacity overlay
            #frame = cv2.addWeighted(frame, 0.5, _frame, 0.5, 0)
            frame = cv2.addWeighted(frame, 0.5, blank_canvas.astype("uint8"), 0.5, 1)

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
