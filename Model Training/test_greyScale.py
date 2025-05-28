import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models

# --------- CONFIG ---------
image_folder = "/Users/emmanuelgeorgep/Documents/Internship/Data/testImage"
model_path = "/Users/emmanuelgeorgep/Documents/Internship/Saved Models/resnet_model.pth"
img_size = (224, 224)  # Change if your model expects a different size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['class1', 'class2']  # Change to your actual class labels
# --------------------------

# Load model (ResNet as example)
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformation for grayscale images
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Predict function
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = class_names[probs.argmax().item()]
        confidence = probs.max().item()
    
    print(f"{os.path.basename(image_path)} → Predicted: {pred_class} ({confidence:.2%})")

# Run predictions on all images in folder
for file in sorted(os.listdir(image_folder)):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        predict_image(os.path.join(image_folder, file))