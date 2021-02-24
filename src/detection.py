"""Detection File

This script provides a collection of detection functions.
"""

import math

import numpy as np
import cv2

from acquisition import VideoStream
from utility_functions import crosshair

# Correction function to project detected point from neck of vial to base
# in order to minimise disparity
def baseCorrection(point, y_object = 0.3, y_camera = 6.0, x_max = 1920, y_max = 1080, strength = 1.8):
    
    # Find centre of camera 
    x_centre = x_max/2
    y_centre = y_max/2

    # Algin the detected point about the central camera point
    x_det = point[0] - x_centre
    y_det = point[1] - y_centre

    # Convert from cartesian to polar coordinates
    r_det = math.sqrt(x_det*x_det + y_det*y_det)
    theta = math.atan2(y_det, x_det)

    # Apply correction to radial component to account for perspective effect
    r_corrected = (r_det * y_camera - r_det * y_object * strength) / y_camera

    # Convert back to cartesian coordinates
    x_corrected = r_corrected * math.cos(theta)
    y_corrected = r_corrected * math.sin(theta)

    # Return point in camera space from centred coordinates
    p_corrected = (int(x_corrected + x_centre), int(y_corrected + y_centre))

    return p_corrected


# Connected components filtering (from segmentation to detection)
def connectedFiltering(frame, mask, debug = False):

    # Use connected components for blob detection
    connectivity = 4
    output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output

    points = []

    # Iterate through detected blobs and extract statistics for filtering
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Extract centroid for given connected component
        (centroid_x, centroid_y) = centroids[i]

        # Perform correction to find the base point
        y_dim, x_dim = frame.shape[:2]
        (corrected_x, corrected_y) = baseCorrection(centroids[i], y_max = y_dim, x_max = x_dim)

        points.append((corrected_x, corrected_y))

        # Perform selection of connection components based on area
        if area > 0 and area < 1500:
            # Draw bounding box for connected component
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Draw connected component centroid (red)
            crosshair(frame, (int(centroid_x), int(centroid_y)), color = (0, 0, 255))

            # Draw corrected base point (green)
            crosshair(frame, (int(corrected_x), int(corrected_y)), color = (0, 255, 0))
        
        # Show detection window if debug enabled
        if debug:
            cv2.imshow("Detections", frame)

    return points


# Hough Circle Detection Function
def houghDetect(frame, dp, minDist, param1, param2, minRadius, maxRadius, debug = False):
    
    # DESCRIPTION OF PARAMETERS ============================================
    # dp is the inverse ratio of accumulator resolution to image resolution
    # Min dist is minimum distance between circle centres
    # Param1 is higher threshold for edge detection
    # Param2 is accumulator threshold
    # Min and max radius constrain the detected circle sizes
    
    # Convert to grayscale for hough circle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Consider blurring to reduce noisy circles
    #gray = cv2.medianBlur(gray,15)

    try:
        
        # Scale dp back to expected range
        dp = dp / 10 if dp > 10 else dp

        # Detect circles using hough circle transform method
        circles = cv2.HoughCircles(gray, method = cv2.HOUGH_GRADIENT,
                               dp = dp, minDist = minDist,
                               param1 = param1, param2 = param2,
                               minRadius = minRadius, maxRadius = maxRadius)
    
        points = []
        try:

            # Convet array to integers, and return x,y,r values
            circles = np.round(circles[0,:].astype("int"))

            for (x,y,r) in circles:
                    # Append circle centres to list
                    points.append((x,y))
                    
                    # Show detection window if debug enabled
                    if debug:
                        cv2.circle(frame, (x,y), int(r), (0,0,255), 2)
                        cv2.imshow("Detections", frame)


        except Exception as e:
            pass

        return True, points
    
    except Exception as e:
        return False, None


# Grayscale Detection Function
def grayscaleDetect(frame):
    pass


# HSV Detection Function
def hsvDetect(frame, hue_low, hue_high, sat_low, sat_high, val_low, val_high, debug = False):
    try:

        # Blur the frame to reduce noise
        frame = cv2.GaussianBlur(frame, (5,5), 0)

        # Convert the colour space of the input frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Convert input parameters into threshold hsv arrays
        hsv_low = np.array([hue_low, sat_low, val_low])
        hsv_high = np.array([hue_high, sat_high, val_high])

        # Mask the frame based on the hsv thresholds
        mask = cv2.inRange(hsv, hsv_low, hsv_high)

        # Segment the frame using the binary mask
        hsv_seg = cv2.bitwise_and(frame, frame, mask = mask)

        points = connectedFiltering(frame, mask, debug)

        return True, points
    
    except:
        return None, False


# Template Match Detection
def templateMatch(frame, placeholder = None, template_path = "Template.png", debug = False):
    try:
    
        # Load template image
        template = cv2.imread(template_path)

        # Extract dimensions
        h, w = template.shape[:2]

        # Select method for template matching
        method = cv2.TM_CCOEFF_NORMED

        # Set match threshold
        threshold = 0.40

        # Perform matching
        matched = cv2.matchTemplate(frame, template, method)

        points = []
        max_val = 1

        # While matched points exist that are above the threshold
        while max_val > threshold:

            # Detect min and max value locations in matched array
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)

            # If value above threshold, annotate frame and append centroid to point list
            if max_val > threshold:
                matched[max_loc[1]-h//2:max_loc[1]+h//2+1, max_loc[0]-w//2:max_loc[0]+w//2+1] = 0   
                cv2.rectangle(frame,(max_loc[0],max_loc[1]), (max_loc[0]+w+1, max_loc[1]+h+1), (0,255,0) )
                points.append((int(max_loc[0]+(w+1)/2), int(max_loc[1] + (h+1)/2)))
        
        # Show the detection window
        cv2.imshow("Detections",frame)

        return True, points
    
    except:
        return False, None


# Perform a sweep to benchmark detectors
def benchmark_detector(frame, detector):
    ret, detections = detector()
    pass


if __name__ == "__main__":
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
    
        # Detect vials

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