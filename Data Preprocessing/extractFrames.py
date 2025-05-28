import cv2
import os

def extract_frames_from_videos(input_folder, output_folder, fps_extraction=1):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video file: {video_file}")
            continue
        
        # Get video fps to calculate frame interval for 1 frame per second
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            print(f"Cannot get FPS for {video_file}, skipping.")
            cap.release()
            continue
        
        frame_interval = int(video_fps // fps_extraction)  # frames to skip between extractions
        
        frame_count = 0
        saved_frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame every frame_interval frames
            if frame_count % frame_interval == 0:
                # Create a unique frame filename
                frame_filename = f"{os.path.splitext(video_file)[0]}_frame_{saved_frame_count:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)
                
                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {saved_frame_count} frames from {video_file}")

if __name__ == "__main__":
    input_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/video"
    output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/temp"
    extract_frames_from_videos(input_folder, output_folder)