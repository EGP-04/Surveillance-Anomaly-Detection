import cv2 # type: ignore #
import os

# --- CONFIGURATION ---
video_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Video/Movie on 12-09-24 at 3.36 PM.mov"
output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/testImage"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if total_frames < 200:
    print(f"⚠️ Video has only {total_frames} frames. Cannot extract 7.")
    cap.release()
    exit()

# Select 100 evenly spaced frames
step = total_frames // 201
frame_indices = [(i + 1) * step for i in range(200)]

# Extract and save frames
for i, idx in enumerate(frame_indices, start=1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()

    if ret:
        out_path = os.path.join(output_folder, f"{i}_new.jpg")
        cv2.imwrite(out_path, frame)
    else:
        print(f"❌ Failed to read frame {idx}")

cap.release()
print("✅ Done extracting 200 frames.")