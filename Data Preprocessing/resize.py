import cv2
import os

def resize_images_in_place(folder_path, size=(320, 240)):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipped unreadable file: {img_file}")
            continue

        resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, resized)
        print(f"Resized and overwritten: {img_path}")

if __name__ == "__main__":
    folder_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Images/Blocked"
    resize_images_in_place(folder_path)