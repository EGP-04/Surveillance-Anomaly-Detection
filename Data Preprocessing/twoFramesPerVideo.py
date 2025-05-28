import cv2
import os

# === Configuration ===
video_folder = '/Users/emmanuelgeorgep/Documents/Internship/Data/Test/Normal'  # Change this to your folder
output_frame_count = 2  # Frames per video

# === Process Each Video ===
for filename in os.listdir(video_folder):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, filename)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < output_frame_count:
            print(f"Video too short: {filename}")
            continue

        # Evenly spaced frame indices
        frame_indices = [int(total_frames * i / output_frame_count) for i in range(output_frame_count)]

        frame_id = 0
        saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id in frame_indices:
                image_filename = f"{os.path.splitext(filename)[0]}_frame{saved+1}.jpg"
                image_path = os.path.join(video_folder, image_filename)
                cv2.imwrite(image_path, frame)
                saved += 1

                if saved >= output_frame_count:
                    break
            frame_id += 1

        cap.release()
        print(f"Extracted {saved} frames from {filename}")