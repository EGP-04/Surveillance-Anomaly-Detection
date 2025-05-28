import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.metrics import classification_report, accuracy_score

# Threshold for classification
THRESHOLD = 25

def compute_mean_8x8_and_std(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_image = uniform_filter(image.astype(np.float32), size=(8, 8))
    stddev = np.std(mean_image)
    return stddev

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

def save_to_csv(filenames, predictions, actuals, output_csv="results.csv"):
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