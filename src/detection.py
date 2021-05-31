"""Detection File

This script provides a collection of detection functions.
"""

import math

import numpy as np
import cv2

from acquisition import VideoStream
from utility_functions import crosshair

CLASS_PLACEHOLDER = 0
PROB_PLACEHOLDER = 0.3

# Correction function to project detected point from neck of vial to base
# in order to minimise disparity
def baseCorrection(point, y_object = 0.3, y_camera = 6.0, 
                          x_max = 1920, y_max = 1080, strength = 1.8):
    
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
    bboxes = []

    debug_frame = frame.copy()

    # Iterate through detected blobs and extract statistics for filtering
    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # Extract centroid for given connected component
        (centroid_x, centroid_y) = centroids[i]

        # Perform selection of connection components based on area
        if area > 100 and area < 1500:

            # Add bounding box to list
            bboxes.append((x, y, x + w, y + h, PROB_PLACEHOLDER, CLASS_PLACEHOLDER))

            if debug:
                # Draw bounding box for connected component
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Draw connected component centroid (red)
                crosshair(debug_frame, (int(centroid_x), int(centroid_y)), color = (0, 0, 255))
        
                # Show detection window if debug enabled
                cv2.imshow("Detections", debug_frame)

    return bboxes, points


# Hough Circle Detection Function
def houghDetect(frame, dp = 1.5, minDist = 20, 
                       param1 = 20, param2 = 20, 
                       minRadius = 10, maxRadius = 15, debug = False):
    
    # DESCRIPTION OF PARAMETERS ============================================
    # dp is the inverse ratio of accumulator resolution to image resolution
    # Min dist is minimum distance between circle centres
    # Param1 is higher threshold for edge detection
    # Param2 is accumulator threshold
    # Min and max radius constrain the detected circle sizes

    debug_frame = frame.copy()
    
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
        bboxes = []

        try:

            # Convet array to integers, and return x,y,r values
            circles = np.round(circles[0,:].astype("int"))

            for (x,y,r) in circles:
                    # Append circle centres to list
                    points.append((x,y))

                    # Add bounding box to list
                    bboxes.append((x-r, y-r, x+r, y+r, PROB_PLACEHOLDER, CLASS_PLACEHOLDER))
                    
                    # Show detection window if debug enabled
                    if debug:
                        cv2.circle(debug_frame, (x,y), int(r), (0,0,255), 2)
                        cv2.imshow("Detections", debug_frame)


        except Exception as e:
            pass

        return True, bboxes, points
    
    except Exception as e:
        return False, None, None


# HSV Detection Function
def hsvDetect(frame, hue_low = 0, hue_high = 179, 
                     sat_low = 0, sat_high = 255, 
                     val_low = 0, val_high = 255, debug = False):
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
        
        if debug:
            cv2.imshow("HSV_mask", hsv_seg)

        bboxes, points = connectedFiltering(frame, mask, debug)

        return True, bboxes, points
    
    except:
        return False, None, None


# Non-Maximal Suppresion Implementation for duplicate bboxes
# Malisiewicz et al.
# Source: PyImageSearch - https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


# Template Match Detection
def templateMatch(frame, match_threshold = 40,
                        template_path_idx = 0, debug = False):
    try:
    
        template_paths = ["Template_New_half.png", "Template_Averaged_half_crop.png"]

        # Load template image
        template = cv2.imread(template_paths[template_path_idx])

        # Extract dimensions
        h, w = template.shape[:2]

        # Select method for template matching
        method = cv2.TM_CCOEFF_NORMED

        # Set match threshold (scaled to match trackbar integer scale)
        threshold = match_threshold/100

        # Perform matching
        matched = cv2.matchTemplate(frame, template, method)

        # Find matched locations
        loc = np.where( matched >= threshold)
        recs = []
        for pt in zip(*loc[::-1]):
            recs.append((pt[0], pt[1], pt[0]+w, pt[1]+h))
        
        # Perform non max suppresion on bboxes, with overlap threshold
        #_bboxes = non_max_suppression_fast(np.asarray(recs), 0.2)
        _bboxes = non_max_suppression_fast(np.asarray(recs), 0.4)
        
        debug_frame = frame.copy()
        points = []
        bboxes = []

        # Annotate frame and append centroids and bboxes to relevant lists
        for _bbox in _bboxes:
            bboxes.append((_bbox[0],_bbox[1],_bbox[2],_bbox[3], PROB_PLACEHOLDER, CLASS_PLACEHOLDER))
            points.append((int(_bbox[0] + (_bbox[2]-_bbox[0])/2), int(_bbox[1] + (_bbox[3]-_bbox[1])/2)))
            if debug:
                    cv2.rectangle(debug_frame, (_bbox[0],_bbox[1]), (_bbox[2],_bbox[3]), (0,255,0), 2)
                    cv2.imshow("Detections",debug_frame)

        return True, bboxes, points
    
    except:
        return False, None, None


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
