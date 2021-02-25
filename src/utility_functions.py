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


# Heron's Formula - for calculating area of a triangle from side lengths
def heron(pt1, pt2, pt3):
    side1 = np.linalg.norm(pt1 - pt2)
    side2 = np.linalg.norm(pt2 - pt3)
    side3 = np.linalg.norm(pt3 - pt1)
    s = 0.5 * (side1 + side2 + side3)
    return np.sqrt(s * (s - side1) * (s - side2) * (s - side3))


# Regularity calculation - find the standard deviation of normalised edge lengths
def regularity(pt1, pt2, pt3):
    side1 = np.linalg.norm(pt1 - pt2)
    side2 = np.linalg.norm(pt2 - pt3)
    side3 = np.linalg.norm(pt3 - pt1)
    sides = np.array([side1, side2, side3])
    norm_sides = sides / np.linalg.norm(sides)
    return np.std(norm_sides)