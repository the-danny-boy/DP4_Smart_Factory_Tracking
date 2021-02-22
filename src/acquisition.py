"""Video Acquisition File

This script provides a class as a wrapper for video acquisition 
from either a webcam or video source. 
"""

import time
import cv2
import numpy as np

class VideoStream(object):
    """
    A class used to represent a Video Stream

    Methods
    -------
    start
        Starts the video stream object
    
    stop
        Stops the video stream object
    
    read
        Fetches the next available frame
    """

    def __init__(self, src, fps = 30, height = 1080, width = 1920, loop = False):
        """
        Parameters
        ----------
        src : int / str
            The video stream source (filepath / camera index)
        fps : int, optional
            The frames to be acquired per second (upper bound)
            (default is 30)
        height : int, optional
            The height of the stream for resizing (default is 1080)
        width : int, optional
            The width of the stream for resizing (default is 1920)
        loop : bool, optional
            Whether or not the video should loop (default is False)
        """
        self.src = src
        self.cap = None
        self.fps = fps
        self.height = height
        self.width = width
        self.loop = loop
        self.isFile = not(str(src).isnumeric())
        
        # Assume a float source is meant to be a webcam (int) input
        if type(self.src) == float:
            self.src = int(self.src)


    def start(self):
        """Starts the video stream with specified properties (if appropriate)."""
        try:
            self.cap = cv2.VideoCapture(self.src)
        except:
            raise FileNotFoundError

        # Set webcam properties if not a video file
        if not self.isFile: 
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)

        else:
            # Placeholder for smoothing, or other operations for video file
            pass


    def stop(self):
        """Stops the video stream and releases stream object."""
        self.cap.release()


    def read(self):
        """Fetches frame data, and a return flag about whether successful."""
        while True:

            # Get return flag and frame information from video capture
            ret, frame = self.cap.read()

            # If end of video and looping, restart the video
            if not ret and self.isFile and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

            # If video, resize to specified frame size (width, height)
            if ret and self.isFile:
                frame = cv2.resize(frame, (self.width, self.height))

            yield ret, frame
        

if __name__ == "__main__":

    # Create a video stream object with video file and specified frame rate and frame size
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
