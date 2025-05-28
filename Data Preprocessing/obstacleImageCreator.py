import cv2
import os
import numpy as np
import random

def rotate_image_and_mask(image, mask, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rotated_mask = cv2.warpAffine(mask, rot_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return rotated_img, rotated_mask

def overlay_obstacles(original_folder, obstacle_folder, output_folder, min_coverage=0.75, max_output=100):
    os.makedirs(output_folder, exist_ok=True)

    original_files = [f for f in os.listdir(original_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    obstacle_files = [f for f in os.listdir(obstacle_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not original_files or not obstacle_files:
        print("No valid images found in folders.")
        return

    random.shuffle(original_files)
    output_count = 0

    for orig_file in original_files:
        if output_count >= max_output:
            break

        orig_path = os.path.join(original_folder, orig_file)
        orig_img = cv2.imread(orig_path)
        if orig_img is None:
            continue

        h, w = orig_img.shape[:2]
        total_area = h * w
        obstacle_placed = False

        # Try multiple obstacle images until one succeeds
        for _ in range(10):
            obs_file = random.choice(obstacle_files)
            obs_path = os.path.join(obstacle_folder, obs_file)
            obs_img = cv2.imread(obs_path, cv2.IMREAD_UNCHANGED)
            if obs_img is None or obs_img.shape[2] < 3:
                continue

            if obs_img.shape[2] == 3:
                alpha = np.ones((obs_img.shape[0], obs_img.shape[1]), dtype=np.uint8) * 255
            else:
                alpha = obs_img[:, :, 3]
                obs_img = obs_img[:, :, :3]

            # Resize based on obstacle's actual non-zero area
            coverage_scale = 1.2  # Start with slightly larger
            target_pixels = min_coverage * total_area

            nonzero_ratio = cv2.countNonZero(alpha) / (alpha.shape[0] * alpha.shape[1])
            scale = np.sqrt(target_pixels / (nonzero_ratio * h * w)) * coverage_scale

            obs_h = min(int(obs_img.shape[0] * scale), h)
            obs_w = min(int(obs_img.shape[1] * scale), w)
            obs_img = cv2.resize(obs_img, (obs_w, obs_h), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (obs_w, obs_h), interpolation=cv2.INTER_NEAREST)

            # Rotate randomly
            angle = random.uniform(0, 360)
            obs_img, alpha = rotate_image_and_mask(obs_img, alpha, angle)

            # Random position
            y_start = random.randint(0, max(h - obs_img.shape[0], 1))
            x_start = random.randint(0, max(w - obs_img.shape[1], 1))

            # Fit within bounds
            obs_img = obs_img[:min(h - y_start, obs_img.shape[0]), :min(w - x_start, obs_img.shape[1])]
            alpha = alpha[:min(h - y_start, alpha.shape[0]), :min(w - x_start, alpha.shape[1])]

            # Overlay
            roi = orig_img[y_start:y_start + obs_img.shape[0], x_start:x_start + obs_img.shape[1]]
            inv_alpha = cv2.bitwise_not(alpha)
            bg = cv2.bitwise_and(roi, roi, mask=inv_alpha)
            fg = cv2.bitwise_and(obs_img, obs_img, mask=alpha)
            combined = cv2.add(bg, fg)
            composite_img = orig_img.copy()
            composite_img[y_start:y_start + combined.shape[0], x_start:x_start + combined.shape[1]] = combined

            # Compute actual coverage
            coverage_mask = np.zeros((h, w), dtype=np.uint8)
            coverage_mask[y_start:y_start + alpha.shape[0], x_start:x_start + alpha.shape[1]] = alpha
            covered_area = cv2.countNonZero(coverage_mask)
            coverage_ratio = covered_area / total_area

            if coverage_ratio >= min_coverage:
                output_path = os.path.join(output_folder, f"obstructed_{output_count+1:03d}.jpg")
                cv2.imwrite(output_path, composite_img)
                print(f"Saved: {output_path} (Coverage: {coverage_ratio:.2%})")
                output_count += 1
                obstacle_placed = True
                break

        if not obstacle_placed:
            print(f"Skipped: {orig_file} - No obstacle could achieve required coverage.")

if __name__ == "__main__":
    original_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/temp_frames"
    obstacle_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/temp_obstacle"
    output_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/temp_obstructed_images"
    overlay_obstacles(original_folder, obstacle_folder, output_folder, max_output=100)