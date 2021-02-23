"""Camera Processing File

This script provides a class as a wrapper for bundled image processing
capabilities to help with scalability.
"""

import cv2
import numpy as np
from acquisition import VideoStream

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

        # 3) Track all vials

        # 4) Analyse frame contents

        # 5) Report summary statistics

        return ret, frame


    # Stop the video stream
    def stop(self):
        self.vs.stop()