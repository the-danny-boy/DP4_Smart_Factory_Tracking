"""Template_Extraction File

This script extracts the RoIs from the training images to create an averaged template image.
"""

import numpy as np
import cv2
import glob
import os

# Path for training data (images, labels)
output_root_directory = r"../Data_Generator/Data_Generator_Outputs/"
image_paths = [os.path.join(output_root_directory, dir_item, "train", "images")+"/" for dir_item in os.listdir(output_root_directory)]
label_paths = [os.path.join(output_root_directory, dir_item, "train", "labels")+"/" for dir_item in os.listdir(output_root_directory)]

# Find all items at target path
images = []
labels = []
for image_path, label_path in zip(image_paths, label_paths):
    images.extend(glob.glob(image_path + "*.jpg"))
    labels.extend(glob.glob(label_path + "*.txt"))

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

# Resize to half resolution
roi = cv2.resize(roi, (int(roi.shape[1] * 0.5), int(roi.shape[0] * 0.5)))

# Now rotate the image to get a mean rotational image
mean_imgs = []
(h, w) = roi.shape[:2]
for theta in range(0, 360, 45):
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, theta, 1)
    rotated = cv2.warpAffine(roi, M, (w, h))
    mean_imgs.append(rotated)

# Find the mean of the images
avg = np.mean(mean_imgs, axis=0)
avg = np.asarray(avg, dtype = "uint8")

# Crop to the region of interest
radius = 15
avg = avg[int(h/2 - radius):int(h/2 + radius), int(w/2 - radius):int(w/2 + radius)]
cv2.imwrite("Template_Averaged_half_crop.png", avg)

# Alert user of completion
print("Done generating averaged template")