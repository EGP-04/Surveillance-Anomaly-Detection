import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models
import time
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.metrics import classification_report, accuracy_score

# Threshold for classification
THRESHOLD = 30

def average_absolute_difference(image):    
    mean_val = np.mean(image)
    abs_diff = np.abs(image.astype(np.float32) - mean_val)
    return np.mean(abs_diff)

def compute_mean_8x8_and_std(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered_image = uniform_filter(image.astype(np.float32), size=(8, 8))  #For sliding window and Mean image
    #mean_image = cv2.GaussianBlur(image.astype(np.float32), (9, 9), 0)  For Gaussian blur
    abs_diff = average_absolute_difference(filtered_image)
    # return abs_diff
    
    '''Croping simple 8X8 image parts and finding '''
    h, w = image.shape
    h_crop = h - h % 8
    w_crop = w - w % 8
    image_cropped = image[:h_crop, :w_crop]

    reshaped = image_cropped.reshape(h_crop // 8, 8, w_crop // 8, 8)
    patch_means = reshaped.mean(axis=(1, 3))
    
    return average_absolute_difference(patch_means)


# Prediction function from webcam frame
def predict_frame(frame):
    ret = compute_mean_8x8_and_std(frame)
    if ret>THRESHOLD:
        return "Normal",(ret-THRESHOLD)
    else:
        return "Blocked",(ret-THRESHOLD)

# Real-time camera prediction
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    fps = 1
    delay = 1 / fps

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            # Convert BGR to RGB for torchvision
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_class, confidence = predict_frame(rgb_frame)

            # Overlay prediction
            label = f"{pred_class} ({confidence})"
            if pred_class=="Normal":
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
                
            cv2.imshow('Real-time Prediction (1 FPS)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()