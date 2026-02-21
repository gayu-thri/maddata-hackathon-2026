import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
"""
# Dataset Structure
```
roads_dataset/
│
├── train/
│   ├── pothole/
│   ├── littering/
│   ├── broken_sign/
│   ├── illegal_parking/
│   ├── damaged_road/
│   └── normal/
│
└── val/
│   ├── pothole/
│   ├── littering/
│   ├── broken_sign/
│   ├── illegal_parking/
│   ├── damaged_road/
│   └── normal/
└── test/
│   ├── pothole/
│   ├── littering/
│   ├── broken_sign/
│   ├── illegal_parking/
│   ├── damaged_road/
│   └── normal/
```
"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder("roads_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder("roads_dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# What model to use? ResNet18 is a good starting point for image classification tasks.
# We can use a pretrained model and fine-tune it on our dataset.
# Other options include VGG, DenseNet, EfficientNet, etc.
model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%")

train_model(model, train_loader, val_loader, epochs=10)

torch.save(model.state_dict(), "roadscan_model.pth")

from PIL import Image

def predict_image(model, image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return train_dataset.classes[predicted.item()]


# Usage example
label = predict_image(model, "test_image.jpg")
print("Prediction:", label)