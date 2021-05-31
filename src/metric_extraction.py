"""Metric Extraction File

This script is used to extract metrics from the video.
"""

import numpy as np
import cv2
from functools import partial
from chrono import Timer
import time
import math
from copy import deepcopy

from acquisition import VideoStream
from utility_functions import crosshair
from YOLO_detector_wrapper import setup, detect_wrapper
from centroid_tracker import Centroid_Tracker

from utility_functions import crosshair, heron, regularity, hsv_to_rgb, map_values, clamp
from itertools import combinations, compress

import scipy.spatial

# Benchmark settings
early_terminate = 246 # Which frame to stop at (and save out analysis)

# Environment Support Geometry
raw_pts = np.asarray([[0,212], [260,487], [404, 486], [400, 327], [410, 328], [959, 147], [959, 538], [0, 537], [0,212]])
contour = np.array(raw_pts[:-1]).reshape((-1,1,2)).astype(np.int32)

# Define subdivision size for void detection
subdiv_distance = 30

# Define criteria for good / ok / bad packing => Void detection
area_ok = 524
area_ok_reg = 519
reg_ok = 0.08
area_bad = 643
area_bad_reg = 563
reg_bad = 0.04

# Define criteria for collision propagation
collision_dist = 50     

# Define video stream
SCALE_FACTOR = 0.5
vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-03-18_20h40m_Camera1_011.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))

# Define detector
model = setup()
detector_func = partial(detect_wrapper, model=model, debug=False)
tracker = Centroid_Tracker(max_lost=3)

# Start the video stream object
vs.start()

frame_no = 0
objectID = 1

times = []
speeds = []
damages = []

# Main logic loop
while True:

    # Acquire next frame
    ret, frame = next(vs.read())
    if not ret:
        break

    # Create duplicate frames for visualising analysis outputs
    collision_frame = frame.copy()
    residence_frame = frame.copy()
    
    # Detect Objects in Frame
    ret, bboxes, points = detector_func(frame)

    # Update Tracker
    objectID, trackedObjects = tracker.update(objectID, points)

    with Timer() as damage_timed:

        population_metrics = {"time":[], "velocity":[], "damage":[]}

        # Iterate through tracked objects and annotate
        for idx, object in zip(trackedObjects.keys(), trackedObjects.values()):

            # Visualise residence time (amount of time in the system) - with continuous colour scale
            residence_time = object["time"]
            visualisation_limit = 180
            hue = 120 - map_values(clamp(residence_time, 0, visualisation_limit), 0, visualisation_limit, 0, 120)
            color = tuple( map( int,hsv_to_rgb((hue, 255, 255))))
            crosshair(residence_frame, object["positions"][-1], size = 8, thick = 4, color = color)

            # Read position deque for tracked object and calculate framewise differences for velocity
            if len(object["positions"]) > 5:
                
                k_scale = 0.01 # Calibration parameter for scaling between image and real world dimensions                
                dt = 1 / 30 # Time increment per frame

                # Find frame-wise motion properties
                positions = np.asarray(object["positions"]) * k_scale
                velocities = np.diff(positions, axis=0) / dt
                mean_speed = np.linalg.norm(np.mean(velocities, axis=0))

                accelerations = np.diff(velocities, axis=0) / dt
                mean_acceleration = np.mean(accelerations, axis=0)
                acceleration_norm = np.linalg.norm(mean_acceleration)

                # High speed (greater changer momentum) collisions contribute to more damage (E=0.5mv^2) => quadratic relation
                # Impacts can cause failure through crack propagation if energetic enough
                # Otherwise cause microcrack formation (which shall be ignored - simplifying assumption)
                damage_speed = 0
                if acceleration_norm > 10:

                    # Increment a damage quantity proportional to squared impact velocity if acceleration norm large enough
                    k_speed = 0.01
                    impact_speed = np.linalg.norm(np.mean(velocities[-4:-1], axis=0))
                    damage_speed = k_speed * impact_speed * impact_speed
                    # Note - coefficient should take into account ability to double up application, or only exert once in collision with wall

                    # Visualise instant collisions
                    crosshair(collision_frame, object["positions"][-1], size = 8, thick = 4, color = (0,0,255)) # red direct / primary

                    # Damage sharing
                    # Extract position of this object and all others (separately)
                    this_pos = object["positions"][-1]
                    other_pos = [trackedObjects[k]["positions"][-1] for k in trackedObjects.keys() if k != idx]

                    # Find the euclidean distance between current point and all others
                    positions_rep = np.tile(this_pos, (len(other_pos),1))
                    norm_diff = np.linalg.norm(other_pos - positions_rep, axis=1)

                    # If close, extract tracker id (key) for incrementing damage
                    collided_instances = norm_diff <= collision_dist
                    indices = list(compress(range(len(collided_instances)), collided_instances))
                    collision_keys = [list(trackedObjects.keys())[_t] for _t in indices]
                    
                    # Add damage equal to collision intensity
                    for collision_key in collision_keys:
                        trackedObjects[collision_key]["damage"] += damage_speed

                        # Visualise other collision targets
                        crosshair(collision_frame, trackedObjects[collision_key]["positions"][-1], size = 8, thick = 4, color = (0,255,255)) # yellow secondary


                # More time in the system increases likeliness of repeated / cumulative loading - incremental
                k_time = 1
                damage_time = k_time * dt

                # Combine damage metrics
                damage_delta = damage_speed + damage_time
                object["damage"] += damage_delta

                # Visualise damaged candidates (red)
                """if object["damage"] > 10:
                    crosshair(frame, object["positions"][-1], size = 8, color = (0,0,255))"""

                population_metrics["time"].append(object["time"])
                population_metrics["velocity"].append(mean_speed)
                population_metrics["damage"].append(object["damage"])

    #print("Damage process time (ms):", damage_timed.elapsed)

    # Log and print population metrics
    """
    times.append(population_metrics["time"])
    speeds.append(population_metrics["velocity"])
    damages.append(population_metrics["damage"])

    print(f'Population residence time: Mean = {np.mean(population_metrics["time"])}, SD = {np.std(population_metrics["time"])}')
    print(f'Population velocity: Mean = {np.mean(population_metrics["velocity"])}, SD = {np.std(population_metrics["velocity"])}')
    print(f'Population damage: Mean = {np.mean(population_metrics["damage"])}, SD = {np.std(population_metrics["damage"])}')
    """


    # VOIDS (SPATIAL DESCRIPTOR)
    # Create canvas for image-based void counting
    blank_canvas = np.zeros(frame.shape)
    with Timer() as void_timed:
        
        # Check if enough (3) points for triangulation
        if len(points) > 3:

            allPt = []

            # Make copy of support contour points
            refPt = deepcopy(raw_pts)

            # Transform to numpy arrays, and create shifted version for distance calcs
            point1 = np.asarray(refPt)
            point2 = np.roll(point1, -1, axis = 0)

            # Iterate over and add points to list (including subdivisions if required)
            for j in range(len(point1)-1):
                
                # Take two adjacent points and calculate the distance between these
                p1 = point1[j]
                p2 = point2[j]
                _p1 = list(map(int, p1))
                allPt.append(_p1)
                dist = np.linalg.norm(p2 - p1)
                
                # Perform subdivision of line segment if too large
                if dist > subdiv_distance:

                    # Find direction vector
                    dir_vector = np.subtract(p2, p1)

                    # Find number of samples (round up to enforce maximum size)
                    sample_no = math.ceil(dist / subdiv_distance)
                    
                    # Calculate subdivision (fragment) distance
                    fragd = subdiv_distance / dist

                    # Iterate through and generate extra points
                    for i in range(sample_no):
                        vecPt = p1 + np.multiply(fragd * i, dir_vector)
                        _vecPt = list(map(int, vecPt))
                        allPt.append(_vecPt)

                # Otherwise add (int) end point to list
                else:
                    _p2 = list(map(int, p2))
                    allPt.append(_p2)

            
            # Extend original point array with support points
            points.extend(allPt)

            # Calculate delaunay triangulation
            ts = scipy.spatial.Delaunay(points)

            # Select triangles
            points = np.asarray(points)
            pts_shaped = points[ts.simplices]

            # Find centre points
            px = np.asarray([(ptt[0][0] + ptt[1][0] + ptt[2][0]) / 3 for ptt in pts_shaped], dtype="int").reshape(-1,1)
            py = np.asarray([(ptt[0][1] + ptt[1][1] + ptt[2][1]) / 3 for ptt in pts_shaped], dtype="int").reshape(-1,1)
            pts = np.concatenate((px,py),axis = 1)

            # Test if centre points in valid region => only keep valid triangle points
            in_contour = [cv2.pointPolygonTest(contour, tuple(pttt), False) for pttt in pts]
            pts = pts[np.asarray(in_contour) < 0]
            pts_shaped = pts_shaped[np.asarray(in_contour) < 0]

            # Combine heron (area) and regularity functions, and map to point list
            comb_funcs = lambda x: (heron(x[0], x[1], x[2]), regularity(x[0], x[1], x[2]))
            comb_outputs = list(map(comb_funcs, pts_shaped))
            ar, reg = map(list, zip(*comb_outputs))

            # Colorise good packing as green (colour all using convex hull)
            conv = points[scipy.spatial.ConvexHull(points).vertices]
            cv2.fillPoly(blank_canvas, [conv], (0,255,0))

            # Overlay / mask invalid region
            cv2.drawContours(blank_canvas,[contour],0,(0,0,0),-1)

            # Filter other severities of packing using logical numpy operator
            warn_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_ok), np.logical_and((np.asarray(ar)>area_ok_reg), (np.asarray(reg) > reg_ok))) ]
            crit_pts = pts_shaped[ np.logical_or((np.asarray(ar)>area_bad), np.logical_and((np.asarray(ar)>area_bad_reg), (np.asarray(reg) > reg_bad))) ]

            # Colorise other regions
            cv2.fillPoly(blank_canvas, warn_pts, (255,0,0))
            cv2.fillPoly(blank_canvas, crit_pts, (0,0,255))

            # Combine adjacent regions and count
            good_canvas = cv2.inRange(blank_canvas, (0,10,0), (0,255,0))
            good_canvas = cv2.erode(good_canvas, (5,5))
            _good_count, _labels, good_stats, good_centroids = cv2.connectedComponentsWithStats(good_canvas)
            good_count = _good_count - 1
                
            crit_canvas = cv2.inRange(blank_canvas, (0,0,10), (0,0,255))
            crit_canvas = cv2.erode(crit_canvas, (5,5))
            _crit_count, _labels, crit_stats, crit_centroids  = cv2.connectedComponentsWithStats(crit_canvas)
            crit_count = _crit_count - 1
                
            warn_canvas = cv2.inRange(blank_canvas, (10,0,0), (255,0,0))
            warn_canvas = cv2.erode(warn_canvas, (5,5))
            _warn_count, _labels, warn_stats, warn_centroids = cv2.connectedComponentsWithStats(warn_canvas)
            warn_count = _warn_count - 1

            # Visualise this as a 50% opacity overlap
            frame = cv2.addWeighted(frame, 0.5, blank_canvas.astype("uint8"), 0.5, 1)

            print("==========================================================================================================================")
            print(f"INFO: {good_count} good region{'s' if good_count>0 else ''}")
            print(f"Location(s): {np.round(good_centroids[1:].astype('int')).tolist()}")
            print(f"Size(s): {good_stats[1:,cv2.CC_STAT_AREA].tolist()}")
                   
            print()
                        
            print(f"INFO: {warn_count} warning region{'s' if warn_count>0 else ''}")
            print(f"Location(s): {np.round(warn_centroids[1:].astype('int')).tolist()}")
            print(f"Size(s): {warn_stats[1:,cv2.CC_STAT_AREA].tolist()}")
            
            print()
            
            print(f"INFO: {crit_count} critical region{'s' if crit_count>0 else ''}")
            print(f"Location(s): {np.round(crit_centroids[1:].astype('int')).tolist()}")
            print(f"Size(s): {crit_stats[1:,cv2.CC_STAT_AREA].tolist()}")
            print("==========================================================================================================================")
            print()


    #print("Void process time (ms):", void_timed.elapsed)
    #print("Total metric time (ms):", damage_timed.elapsed + void_timed.elapsed)

    """
    for p in points:
        crosshair(frame, p, size = 8, color = (0,0,0))
    """

    # Show annotated frame
    cv2.imshow("Frame", frame)
    cv2.imshow("Collision_Frame", collision_frame)
    cv2.imshow("Residence_Frame", residence_frame)
    
    # Only wait for 1ms to limit the performance overhead
    key = cv2.waitKey(1) & 0xFF

    # Escape / close if "q" pressed
    if key == ord("q"):
        break

    # Pause on "p" key
    if key == ord("p"):
        cv2.waitKey(0) & 0xFF

    # Early termination and output generation
    if frame_no == early_terminate:
        cv2.imwrite("Analysis_Collision.png", collision_frame)
        cv2.imwrite("Analysis_Cumulative.png", residence_frame)
        cv2.imwrite("Analysis_Toppling.png", frame)
        break
    
    frame_no += 1

# Tidy up - close windows and stop video stream object
cv2.destroyAllWindows()
vs.stop()