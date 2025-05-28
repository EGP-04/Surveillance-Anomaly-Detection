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

def process_folder(folder_path):
    predictions = []
    actuals = []
    filenames = []

    for class_name in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            file_path = os.path.join(class_path, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue

            std_value = compute_mean_8x8_and_std(image)
            prediction = "Normal" if std_value >= THRESHOLD else "Blocked"

            predictions.append(prediction)
            actuals.append(class_name)
            filenames.append(filename)

            print(f"{filename} → STD: {std_value:.2f} → Predicted: {prediction}, Actual: {class_name}")

    return filenames, predictions, actuals

def save_to_csv(filenames, predictions, actuals, output_csv="results_imgProcess.csv"):
    df = pd.DataFrame({
        "Filename": filenames,
        "Prediction": predictions,
        "Actual": actuals
    })
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

def evaluate(predictions, actuals):
    print("\nClassification Report:")
    print(classification_report(actuals, predictions))
    print(f"Accuracy: {accuracy_score(actuals, predictions):.2%}")

if __name__ == "__main__":
    folder_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Test"  # Replace with your path

    filenames, predictions, actuals = process_folder(folder_path)
    save_to_csv(filenames, predictions, actuals)
    evaluate(predictions, actuals)