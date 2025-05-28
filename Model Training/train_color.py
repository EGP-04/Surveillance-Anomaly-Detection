import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

# === Configuration ===
data_dir = '/Users/emmanuelgeorgep/Documents/Internship/Data/Images'
output_dir = '/Users/emmanuelgeorgep/Documents/Internship/Saved Models/ColorWithFace'
os.makedirs(output_dir, exist_ok=True)

image_size = (224, 224)
batch_size = 32
num_classes = 2
num_epochs = 5
test_ratio = 1 / 11  # 1 test for every 10 train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Load full dataset ===
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# === Train-test split ===
total_size = len(full_dataset)
test_size = int(test_ratio * total_size)
train_size = total_size - test_size

train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Model factory ===
def get_model(name):
    if name == 'resnet':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == 'densenet':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == 'vgg':
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Unknown model name")
    return model.to(device)

# === Training and testing ===
def train_and_evaluate(model, train_loader, test_loader, model_name, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_train, correct_train = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = 100 * correct_train / total_train

        # === Evaluation ===
        model.eval()
        total_test, correct_test = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

        test_acc = 100 * correct_test / total_test

        print(f"[{model_name.upper()}] Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}%")

    # Save model
    model_path = os.path.join(output_dir, f"{model_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

# === Run training for each model ===
for model_name in ['resnet', 'efficientnet', 'mobilenet', 'densenet', 'vgg']:
    print(f"\n=== Training {model_name.upper()} ===")
    model = get_model(model_name)
    train_and_evaluate(model, train_loader, test_loader, model_name, num_epochs)
    
    
    
    
