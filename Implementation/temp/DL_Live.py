import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models
import time

# --------- CONFIG ---------
model_path = "/Users/emmanuelgeorgep/Documents/Internship/Saved Models/ColorWithFace/resnet_model.pth"
img_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Blocked', 'Normal']  # Replace with your actual classes
# --------------------------

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformation for color images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Prediction function from webcam frame
def predict_frame(frame):
    img_tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = class_names[probs.argmax().item()]
        confidence = probs.max().item()
    return pred_class, confidence

# Real-time camera prediction
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    fps = 1
    delay = 1 / fps

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            # Convert BGR to RGB for torchvision
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred_class, confidence = predict_frame(rgb_frame)

            # Overlay prediction
            label = f"{pred_class} ({confidence:.1%})"
            if pred_class=="Normal":
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
                
            cv2.imshow('Real-time Prediction (1 FPS)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()