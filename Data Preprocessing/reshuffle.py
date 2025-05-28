import os
import random

# --- CONFIGURATION ---
folder_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Images/Normal"
image_extensions = ('.jpg', '.jpeg', '.png')  # Allowed image types

# Get all images named as numbers and with proper extension
images = [f for f in os.listdir(folder_path)
          if f.lower().endswith(image_extensions) and f.split('.')[0].isdigit()]

# Sort numerically and shuffle
images.sort(key=lambda x: int(x.split('.')[0]))
random.shuffle(images)

# Phase 1: Rename all images to unique temporary names
temp_map = {}
for idx, old_name in enumerate(images):
    old_path = os.path.join(folder_path, old_name)
    temp_name = f"__tempfile_{idx}.tmp"
    temp_path = os.path.join(folder_path, temp_name)
    os.rename(old_path, temp_path)
    temp_map[temp_name] = idx + 1  

# Phase 2: Rename from temporary to final names
for temp_name, new_index in temp_map.items():
    temp_path = os.path.join(folder_path, temp_name)
    new_name = f"{new_index}.jpg"  
    new_path = os.path.join(folder_path, new_name)
    os.rename(temp_path, new_path)

print(f"✅ Successfully reshuffled and renamed {len(images)} images.")