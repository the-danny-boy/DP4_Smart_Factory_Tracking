"""Template_Extraction File

This script extracts the RoIs from the training images to create an averaged template image.
"""

import numpy as np
import cv2
import glob

# Path for training data (images, labels)
image_path = r"ScaledYOLOv4/vials/train/images/"
label_path = r"ScaledYOLOv4/vials/train/labels/"

# Find all items at target path
images = glob.glob(image_path + "*.jpg")
labels = glob.glob(label_path + "*.txt")

"""
# Extract numeric ids for consistent sorting
images_idx = [int(image[:-4]) for image in images]
labels_idx = [int(label[:-4]) for label in labels]
"""

# Better method - sort alphabetically
# Not numerically increasing, but is consistent
images = sorted(images)
labels = sorted(labels)

# Create empty list for storing RoI images
rois = []

for img_file, dets_file in zip(images, labels):
    
    # Read the image
    img = cv2.imread(img_file)
    
    # Read label file
    with open(dets_file) as f:

        # Read all lines of label file
        dets = f.readlines()
        
        # Iterate through lines in the label file
        for det in dets:
            
            # Extract bbox information
            obj_class, x, y, w, h = list(map(float, det.split(" ")))

            # Undo normalisation of bbox
            x = int(x * img.shape[1])
            w = int(w * img.shape[1])
            y = int(y * img.shape[0])
            h = int(h * img.shape[0])

            # Check if entirely enclosed within frame
            if y-h>0 and x-w>0 and y+h < img.shape[0] and x+w < img.shape[1]:

                # If so, extract ROI and append to list
                roi = img[y-h:y+h, x-w:x+w]
                rois.append(roi)

# Calculate the mean of the ROI images, and convert to integer values
roi = np.mean(rois, axis=0)
roi = np.asarray(roi, dtype = "uint8")

# Write out averaged image, and inform user or completion
cv2.imwrite("Template_Averaged.png", roi)
print("Done generating averaged template")