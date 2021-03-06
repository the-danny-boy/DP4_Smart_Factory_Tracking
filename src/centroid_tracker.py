"""Centroid Tracker File

This script provides the definition for the Centroid Tracker class.
"""

import cv2
import numpy as np

from acquisition import VideoStream
from detection import houghDetect, baseCorrection
from utility_functions import crosshair

from functools import partial
from collections import deque


class Centroid_Tracker(object):
    """
    A class used for the custom Centroid_Tracker object.

    Methods
    -------
    register
        Add a new tracked object to the tracker

    deregister
        Remove a tracked object from the tracker

    update
        Updates the centroid tracker with the current detections

    
    Attributes
    ----------
    objects : dict
        Dictionary storing the tracked objects and their data, [positions, time, damage]

    dropped : dict
        Dictionary storing the number of subsequent frames a detection is lost for
    
    max_lost : int
        Number of frames a detection can be lost for before dropping the tracked object

    seed : int
        Tracker ID generation seed (e.g. used for random number / UUID, or starting increment)

    """

    def __init__(self, max_lost = 3, seed = 0):
        """
        Parameters
        ----------
        max_lost : int
            Number of missed associations before dropping track

        seed : int
            Numeric seed used to designate starting integer
        """

        self.objects = {}
        self.dropped = {} 
        self.max_lost = max_lost
        self.seed = seed


    def register(self, objectID, centroid, time = 0, damage = 0.0, new = True):
        """
        Parameters
        ----------
        objectID : int
            (Numeric) ID to be associated to current object

        centroid : tuple (int,int)
            Tuple storing the location of the object in (local) image space
        
        time : int
            Time value to be assigned to the detector (how long in the system)

        damage : float
            Number representing the amount of damage (based on acceleration history)
        
        new : bool
            Flag indicating whether a new (zero data) tracker assignment or existing insertion
        """

        self.objects[objectID] = {"positions" : deque([centroid]), "time":time, "damage":damage}
        self.dropped[objectID] = 0

        # If new assignment, increment the objectID counter
        if new:
            objectID += 1
            return objectID

    
    def deregister(self, objectID):
        """
        Parameters
        ----------
        objectID : int
            (Numeric) ID of the object to be deregistered
        """

        del self.objects[objectID]
        del self.dropped[objectID]


    def update(self, points, objectID):
        """
        Parameters
        ----------
        points : list
            List of detected points
        
        objeectID : int
            (Numeric) ID to be assigned next
        """
        
        # 1) Get all detected points
        points = np.asarray(points)

        # 2) Check number, and proceed accordingly

        # 3) If none detected, decrement all trackers
        if len(points) == 0:

            # Iterate through objectIDs and increment drop count
            for objectID in list(self.dropped.keys()):
                self.dropped[objectID] += 1

                # If exceeding maximum drop count, deregister
                if self.dropped[objectID] > self.max_lost:
                    self.deregister(objectID)

            return objectID, self.objects


        # 4) If detected, but none tracked, register all
        if len(self.objects) == 0:
            for point in points:
                objectID = self.register(objectID, point)

        # 5) If detected and tracked, associate with previous (linear sum assignment / bipartite minimum weight matching problem)
        #    Register / deregister the remaining points that aren't associated with previous detections


        # Return objectID and tracked objects (with associated data)
        return objectID, self.objects

 

    
if __name__ == "__main__":

    # Create and start video stream
    SCALE_FACTOR = 0.5
    vs = VideoStream(src = "../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm", 
                    fps = 30, height = int(1080*SCALE_FACTOR), width = int(1920*SCALE_FACTOR))
    vs.start()

    # Define detector
    detector_func = partial(houghDetect, dp = 1.5, minDist = 20, 
                                     param1 = 27, param2 = 19, 
                                     minRadius = 12, maxRadius = 15, debug = False)

    # Create Tracker
    tracker = Centroid_Tracker(max_lost = 3, seed = 0)
    objectID = 1

    while True:
        # Process next frame
        ret, frame = next(vs.read())
        if not ret:
            break

        ret, bboxes, points = detector_func(frame = frame)

        baseCorrectionFunc = partial(baseCorrection, y_object = 0.3, 
                y_camera = 6.0, x_max = frame.shape[1], y_max = frame.shape[0], strength = 1.8)

        # Apply baseCorrection to all detected points
        corrected_points = list(map(baseCorrectionFunc, points))

        # Placeholder tracker interface
        # Input points, ID => output ID, objects
        objectID, trackedObjects = tracker.update(corrected_points, objectID)
        print(trackedObjects)

        if ret:
            for point in corrected_points:
                crosshair(frame, point, size = 8, color = (0,0,255))

        # Display frame contents
        cv2.imshow("Frame", frame)

        # Wait 16ms for user input - simulating 60FPS
        # Note - could synchronise with timer and moving average for more control
        key = cv2.waitKey(16) & 0xFF

        # Escape / close if "q" pressed
        if key == ord("q"):
            break

    vs.stop()
    cv2.destroyAllWindows()