"""Utility Functions File

This script provides a collection of utility functions for common use.
"""

import numpy as np
import cv2

# Crosshair visualisation function - draw crosshair at target location
def crosshair(frame, pt, size = 4, thick = 2, color = (0,0,255)):
    x, y = list(map(int, pt))
    cv2.line(frame, (x-size, y), (x+size, y), color, thick)
    cv2.line(frame, (x, y-size), (x, y+size), color, thick)