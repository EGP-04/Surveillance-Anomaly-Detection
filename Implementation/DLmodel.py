import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

# ====== Config ======
data_dir = "/Users/emmanuelgeorgep/Documents/Internship/Data/Test"  # this folder contains class subfolders
model_path = "/Users/emmanuelgeorgep/Documents/Internship/Saved Models/ColorWithFace/resnet_model.pth"
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Transforms ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ====== Load Dataset ======
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ====== Load Model ======
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ====== Predict and Evaluate ======
all_preds = []
all_labels = []
all_filenames = []

with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_filenames.extend([dataset.imgs[i][0] for i in range(len(all_labels) - len(labels), len(all_labels))])

# ====== Save CSV ======
df = pd.DataFrame({
    "filename": [os.path.basename(f) for f in all_filenames],
    "actual": [class_names[i] for i in all_labels],
    "predicted": [class_names[i] for i in all_preds]
})
df.to_csv("predictions_DL.csv", index=False)
print("CSV saved as predictions.csv")

# Extract the ground truth and predicted labels
y_true = df["actual"]
y_pred = df["predicted"]

# Generate classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred))

# Optional: print accuracy
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")