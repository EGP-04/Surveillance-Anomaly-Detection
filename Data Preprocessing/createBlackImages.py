import cv2
import numpy as np
import os

def create_black_images(output_folder, image_size=(320, 240), count=5):
    os.makedirs(output_folder, exist_ok=True)
    
    for i in range(count):
        black_img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        filename = os.path.join(output_folder, f"black_{i+1:02d}.jpg")
        cv2.imwrite(filename, black_img)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/color_obstacles"
    create_black_images(output_folder)