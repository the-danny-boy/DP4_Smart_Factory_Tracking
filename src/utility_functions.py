"""Utility Functions File

This script provides a collection of utility functions for common use.
"""

import math
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


# Maps a value from source to target domain
def map_values(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min


# Clamps a value between a minimum and maximum value
def clamp(x, min_x, max_x):
    return max(min_x,min(x, max_x))


# Converts a given hsv value to rgb
def hsv_to_rgb(hsv):
    (h,s,v) = hsv

    #Normalise
    h = h / 360
    s = s / 255
    v = v / 255

    #Conversion maths
    i = math.floor(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    # Conditional results
    if i % 6 == 0:
        r = v * 255
        g = t * 255
        b = p * 255
    elif i % 6 == 1:
        r = q * 255
        g = v * 255
        b = p * 255
    elif i % 6 == 2:
        r = p * 255
        g = v * 255
        b = t * 255
    elif i % 6 == 3:
        r = p * 255
        g = q * 255
        b = v * 255
    elif i % 6 == 4:
        r = t * 255
        g = p * 255
        b = v * 255
    elif i % 6 == 5:
        r = v * 255
        g = p * 255
        b = q * 255

    #Return as bgr colour (for OpenCV compatibility)
    return (int(b),int(g),int(r))