import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_std_for_images(folder_path):
    class_std = {}
    
    # List subdirectories
    classes = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    for class_name in classes:
        class_folder = os.path.join(folder_path, class_name)
        std_values = []

        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            print(filename)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, (448, 448))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Compute standard deviation
                std_val = np.std(img)
                std_values.append(std_val)
        
        class_std[class_name] = std_values
    
    return class_std

def find_best_threshold(class_std):
    class_names = list(class_std.keys())
    mean1 = np.mean(class_std[class_names[0]])
    mean2 = np.mean(class_std[class_names[1]])
    threshold = (mean1 + mean2) / 2
    return threshold

def plot_std_with_threshold(class_std, threshold):
    plt.figure(figsize=(10, 6))
    
    for class_name, std_vals in class_std.items():
        plt.plot(range(1, len(std_vals)+1), std_vals, label=class_name, marker='o')
    
    # Plot the separation threshold
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    
    plt.title("Standard Deviation of Grayscale Images with Separation Threshold")
    plt.xlabel("Serial Number")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
folder_path = '/Users/emmanuelgeorgep/Documents/Internship/Data/Images'  # Replace with your folder path
class_std = compute_std_for_images(folder_path)
threshold = find_best_threshold(class_std)
plot_std_with_threshold(class_std, threshold)