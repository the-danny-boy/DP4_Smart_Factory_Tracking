"""Camera Processing File

This script provides a class as a wrapper for bundled image processing
capabilities to help with scalability.
"""

import cv2
import numpy as np
from acquisition import VideoStream
from utility_functions import crosshair, heron, regularity
from detection import baseCorrection
from functools import partial
from itertools import combinations

class StreamProcessor:

    def __init__(self, src = None, detector = None, tracker = None):
        self.src = src
        self.detector = detector
        self.tracker = tracker

        # If src is video stream, use this directly
        if type(src) == VideoStream:
            self.vs = src

        # Else, instantiate new video stream with default properties
        else:
            self.vs = VideoStream(self.src)
        
        self.vs.start()
    

    # Function to update the stream processor with the next frame contents
    # Encapsulates all individual image processing components (detect, track, etc.)
    def update(self, analyse = False):

        # 1) Acquires next frame
        ret, frame = next(self.vs.read())
        if not ret:
            return False, None
        

        # 2) Detect all vials in frame
        ret, points = self.detector(frame = frame)

        # 2.1) Draw on the (original) detections - red
        for point in points:
            crosshair(frame, point, size = 8, color = (0,0,255))
        
        # Define partial correction function for mapped application
        baseCorrectionFunc = partial(baseCorrection, y_object = 0.3, 
                y_camera = 6.0, x_max = frame.shape[1], y_max = frame.shape[0], strength = 1.8)

        # Apply baseCorrection to all detected points
        corrected_points = list(map(baseCorrectionFunc, points))

        # 2.2) Draw on the (corrected / base) detections - green
        for corrected_point in corrected_points:
            crosshair(frame, corrected_point, size = 8, color = (0,255,0))


        # 3) Track all vials


        # 4) Analyse frame contents
        if analyse:

            # Motion Metrics - Position, Speed


            # Collision detection + damage contribution (proportional to speed)


            # Void / Pack Spacing Characterisation 
            # => Delaunay Triangulation for Lattice Representation

            # Create a subdiv using the frame geometry
            rect = (0,0,frame.shape[1],frame.shape[0])
            subdiv = cv2.Subdiv2D(rect)

            # Note: using corrected points AS the points list for analysis(?)
            #points = corrected_points

            # Insert detected points into the subdiv, and fetch triangulation
            for p in points:
                subdiv.insert(p)
            ts = subdiv.getTriangleList()

            # Reshape the triangle list into matrix of point triples
            pts_shaped = np.asarray(ts).reshape(-1,3,2)
            for pts in pts_shaped:

                # Find the triangle area
                ar = heron(pts[0], pts[1], pts[2])

                # Find the triangle regularity
                reg = regularity(pts[0], pts[1], pts[2])

                # Explicit void metric is combination of regularity and size
                # => Check if too large or skewed (likely indicator of bad void)
                if (reg > 0.1 and ar > 500) or ar > 1000:

                    # Iterate through all pairs and draw lines
                    for pt1, pt2 in combinations(pts, 2):
                        cv2.line(frame, tuple(pt1), tuple(pt2), (255,0,0), 4)

                    # Draw a filled polygon to highlight the void
                    pts = np.asarray(pts, dtype=np.int32)               
                    cv2.fillPoly(frame, [pts], (0,0,255))                    

            # Display analysis output
            cv2.imshow("Frame", frame)

        # 5) Report summary statistics

        return ret, frame


    # Stop the video stream
    def stop(self):
        self.vs.stop()