"""Centroid Tracker File

This script provides the definition for the Centroid Tracker class.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
import lap

from acquisition import VideoStream
from detection import houghDetect, baseCorrection
from utility_functions import crosshair

from functools import partial
from collections import deque
from chrono import Timer


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

    def __init__(self, max_lost = 3, seed = 0, max_samples = 5):
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
        self.max_samples = max_samples


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


    def update(self, objectID, points):
        """
        Parameters
        ----------
        objectID : int
            (Numeric) ID to be assigned next

        points : list
            List of detected points
        """
        
        # Get detected points
        points = np.asarray(points)

        # If none detected, decrement all trackers
        if len(points) == 0:

            # Iterate through objectIDs and increment drop count
            for objectID in list(self.dropped.keys()):
                self.dropped[objectID] += 1

                # If exceeding maximum drop count, deregister
                if self.dropped[objectID] > self.max_lost:
                    self.deregister(objectID)

            return objectID, self.objects


        # If detected, but none tracked, register all
        elif len(self.objects) == 0:
            for point in points:
                objectID = self.register(objectID, point)


        # Else, perform point association / matching (linear sum assignment)
        # Add / remove trackers as appropriate based on number
        else:

            # Fetch point locations and ids
            detected_pts = points
            tracked_ids = list(self.objects.keys())
            tracked_pts = [_object["positions"][-1] for _object in self.objects.values()]

            #Find matches between points and tracked points       
            cost_matrix = cdist(detected_pts, tracked_pts)
            cost, x, y = lap.lapjv(cost_matrix, extend_cost=True)

            # This gives the closest detected index for each tracked index
            # Result in the form of [[detector_index], [tracker_index]]
            matched_indices = np.array([[y[i],i] for i in x if i >= 0])

            # Get unmatched detector indices (those not present in match result)
            unmatched_detections = []
            for d, detection in enumerate(detected_pts):
                if d not in matched_indices[:, 0]:
                    unmatched_detections.append(d)
                
            # Get unmatched tracker indices (those not present in match result)
            unmatched_trackers = []
            for d, detection in enumerate(tracked_pts):
                if d not in matched_indices[:, 1]:
                    unmatched_trackers.append(d)
            
            # Format match indices as required
            # NOTE - could penalise or constrain based on cost
            matches = []
            for m in matched_indices:
                matches.append(m.reshape(1,2))

            # Populate the numpy match array using current match data
            if(len(matches)==0):
                matches = np.empty((0,2),dtype=int)
            else:
                matches = np.concatenate(matches,axis=0)

            # Assign / update trackers for matching associations
            for m in matches:

                # Get ids (and positoin) from match array
                detected_pt_idx = m[0]
                tracker_pt_idx = m[1]
                detected_pt_position = detected_pts[detected_pt_idx]

                # Get current tracked object values
                _positions = self.objects[tracked_ids[tracker_pt_idx]]["positions"]
                _time = self.objects[tracked_ids[tracker_pt_idx]]["time"]
                _damage = self.objects[tracked_ids[tracker_pt_idx]]["damage"]

                # Update the position deque (limit to length = self.max_samples)
                max_samples = 5
                if len(_positions) > self.max_samples:
                    _positions.popleft()
                _positions.append(detected_pt_position)

                # Increment time counter
                _time += 1

                # TODO - implement damage accumulation

                # Assign values to corresponding tracker
                self.objects[tracked_ids[tracker_pt_idx]] = \
                    {"positions" : _positions, "time": _time, "damage": _damage}

                # Reset drop flag
                self.dropped[tracked_ids[tracker_pt_idx]] = 0


            # For each unmatched detection point (no corresponding point)
            # Register as new tracked object
            for m in unmatched_detections:
                objectID = self.register(objectID, detected_pts[m])

            # For each unmatched (lost) tracker, decrement
            for m in unmatched_trackers:
                tracker_pt_idx = tracked_ids[m]
                self.dropped[tracker_pt_idx] += 1

                # If exceeding maximum drop count, deregister
                if self.dropped[tracker_pt_idx] > self.max_lost:
                    self.deregister(tracker_pt_idx)


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

        with Timer() as timed:
        
            # Process next frame
            ret, frame = next(vs.read())
            if not ret:
                break

            ret, bboxes, points = detector_func(frame = frame)

            baseCorrectionFunc = partial(baseCorrection, y_object = 0.3, 
                    y_camera = 6.0, x_max = frame.shape[1], y_max = frame.shape[0], strength = 1.8)

            # Apply baseCorrection to all detected points
            corrected_points = list(map(baseCorrectionFunc, points))

            # Update tracker
            objectID, trackedObjects = tracker.update(objectID, corrected_points)

            # Iterate through tracked objects
            for idx, object in zip(trackedObjects.keys(), trackedObjects.values()):

                # Draw crosshair
                crosshair(frame, object["positions"][-1], size = 8, color = (0,0,255))

                # Assign label for object id
                text = str(idx)
                font = cv2.FONT_HERSHEY_PLAIN
                fontScale = 1
                (text_width, text_height) = cv2.getTextSize(text, font, fontScale = fontScale, thickness = 2)[0]
                text_offset_x = object["positions"][-1][0]
                text_offset_y = object["positions"][-1][1] - 10
                cv2.putText(frame, text, (int(text_offset_x - text_width/2), int(text_offset_y + text_height/2)), 
                        font, fontScale = fontScale, color = (0,255,0), thickness = 2)

            """if ret:
                for point in corrected_points:
                    crosshair(frame, point, size = 8, color = (0,0,255))"""


            # Display frame contents
            cv2.imshow("Frame", frame)

            # Wait 16ms for user input - simulating 60FPS
            # Note - could synchronise with timer and moving average for more control
            key = cv2.waitKey(16) & 0xFF

            # Escape / close if "q" pressed
            if key == ord("q"):
                break
        
        print("Elapsed Time:", timed.elapsed)

    vs.stop()
    cv2.destroyAllWindows()