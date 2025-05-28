import os

def safe_rename_images(folder_path, extension='.jpg'):
    # Get all image files
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Step 1: Rename to temporary names to avoid collisions
    temp_names = []
    for i, old_name in enumerate(image_files):
        temp_name = f"__temp__{i:04d}.tmp"
        old_path = os.path.join(folder_path, old_name)
        temp_path = os.path.join(folder_path, temp_name)
        os.rename(old_path, temp_path)
        temp_names.append(temp_path)

    # Step 2: Rename to final numerical names
    for i, temp_path in enumerate(temp_names, start=1):
        final_name = f"{i}.jpg"
        final_path = os.path.join(folder_path, final_name)
        os.rename(temp_path, final_path)
        print(f"Renamed to: {final_path}")

if __name__ == "__main__":
    folder_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Images/Blocked"
    safe_rename_images(folder_path)