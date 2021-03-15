import cv2
import numpy as np

from models.experimental import attempt_load
import torch

from utils.torch_utils import select_device
from utils.general import (check_img_size, non_max_suppression, scale_coords)
import random
from chrono import Timer

from utils.datasets import letterbox


def setup():
    # Setup params
    device = select_device("")
    imgsz = 416
    weights = "runs/exp5_yolov4-csp-orig-1k-200/weights/best_yolov4-csp-orig-1k-200_strip.pt"

    # Load an initial model and use this to warm up
    model = attempt_load(weights, map_location = device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    _img = torch.zeros((1, 3, imgsz, imgsz), device=device)
    _ = model(_img)

    return model


def detect(frame, model):

    # Setup params
    imgsz = 416
    device = select_device("")
    augment = False
    half = False
    agnostic_nms = False
    classes = None
    conf_thres = 0.5
    iou_thres = 0.4

    # Resize the input image
    img = letterbox(frame, new_shape=imgsz)[0]

    # Convert to compatible array
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    # Send to GPU (and convert to half precision if valid)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()

    # Normalise / rescale image
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Perform inference
    pred = model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections (and resize to frame)
    for i, det in enumerate(pred):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

    return pred


if __name__ == "__main__":

    # Acquire video footage
    cap = cv2.VideoCapture("../../Data_Generator/Assets/Outputs/2021-02-21_23h37m_Camera1_005.webm")
    
    # Get initial frame
    _, frame = cap.read()

    # Setup model
    model = setup()

    # Main logic loop
    while True:

        # Acquire next frame
        ret, frame = cap.read()

        # Break if failed to acquire frame
        if not ret:
                break

        # Begin detection timer
        with Timer() as timed:

            # Perform detection function without gradient calc (inference)
            with torch.no_grad():
                detections = detect(frame, model)
            
            # Iterate through detections to find the centroids and draw these
            points = []
            if detections:
                for d in detections:
                    if d is not None:
                        for b in d:
                            x1, y1, x2, y2, conf, class_no = b.cpu()
                            centroid = tuple([int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2)])

                            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                            cv2.drawMarker(frame, centroid, color=(0,0,255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=4)
                            
                            points.append(centroid)

        # Print the elapsed time for detection
        print("Elapsed time is:", timed.elapsed)

        # Display the annotated frame
        cv2.imshow("Frame", frame)

        # Check for key press to break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # Tidy up - destroy windows and release video
    cv2.destroyAllWindows()
    cap.release()