import cv2
import numpy as np
import os

def generate_distinct_color_images(output_folder, image_size=(320, 240), count=42):
    os.makedirs(output_folder, exist_ok=True)

    for i in range(count):
        # Generate a distinct hue value
        hue = int((i * 180 / count) % 180)  # OpenCV hue range: 0–179
        hsv_color = np.full((image_size[1], image_size[0], 3), (hue, 255, 255), dtype=np.uint8)  # full saturation, full brightness
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)

        filename = os.path.join(output_folder, f"{i+1:04d}.jpg")
        cv2.imwrite(filename, bgr_color)
        print(f"Saved: {filename}")

if __name__ == "__main__":
    output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/color_obstacles"
    generate_distinct_color_images(output_folder)