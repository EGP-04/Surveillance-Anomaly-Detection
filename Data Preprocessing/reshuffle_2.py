import os

def rename_images_numerically(folder_path, prefix='', extension='.jpg', padding=4):
    image_files = sorted([f for f in os.listdir(folder_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, old_name in enumerate(image_files, start=1):
        new_name = f"{prefix}{str(i).zfill(padding)}{extension}"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

if __name__ == "__main__":
    folder_path = "/Users/emmanuelgeorgep/Documents/Internship/Data/Images/Blocked"
    rename_images_numerically(folder_path)