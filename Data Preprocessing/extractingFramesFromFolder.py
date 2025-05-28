import cv2 # type: ignore
import os
import glob

# --- CONFIGURATION ---
video_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/Videos"           # Folder containing your 110 videos
output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/temp"   # Folder to save the 550 extracted frames

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get all video files, ignoring .DS_Store or hidden files
video_files = [
    f for f in glob.glob(os.path.join(video_folder, "*"))
    if not os.path.basename(f).startswith('.') and os.path.isfile(f)
]

video_files.sort()  # Optional: for consistent ordering

frame_counter = 551  # For naming output images 1.jpg to 550.jpg

# Process each video
for video_path in video_files:
    if frame_counter > 5000:
        break

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < 5:
        print(f"⚠️ Skipping {os.path.basename(video_path)}: not enough frames ({total_frames})")
        cap.release()
        continue

    # Get 5 evenly spaced frame indices
    step = total_frames // 31
    frame_indices = [(i + 1) * step for i in range(30)]

    for idx in frame_indices:
        if frame_counter > 5005:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if ret:
            out_path = os.path.join(output_folder, f"{frame_counter}.jpg")
            cv2.imwrite(out_path, frame)
            frame_counter += 1
        else:
            print(f"❌ Could not read frame {idx} from {os.path.basename(video_path)}")

    cap.release()

print(f"✅ Done. Extracted {frame_counter - 1} frames.")