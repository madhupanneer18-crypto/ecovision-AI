"""
EcoVision AI - Multi-Sample Waste Classifier Fine-Tuning
Trains a deep learning waste classifier on diverse real-world waste images with PyTorch and torchvision.
Outputs calibrated PyTorch weights: models/waste_classifier.pth
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TRAIN_DIR = BASE_DIR / "data" / "train_dataset"

CLASSES = ["Plastic", "Glass", "Metal", "Paper", "Cardboard", "Trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

class DiverseWasteDataset(Dataset):
    def __init__(self, repeats_per_image=35):
        self.items = []
        self.transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=25),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for cls_name in CLASSES:
            folder = TRAIN_DIR / cls_name
            label_idx = CLASS_TO_IDX[cls_name]
            if folder.exists():
                img_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                for img_path in img_files:
                    try:
                        base_img = Image.open(img_path).convert("RGB")
                        for _ in range(repeats_per_image):
                            self.items.append((base_img, label_idx))
                    except Exception as e:
                        print(f"Skipping bad image {img_path}: {e}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base_img, label = self.items[idx]
        tensor_img = self.transform(base_img)
        return tensor_img, label


def train_waste_model():
    print("Preparing Diverse Waste Classifier dataset...")
    dataset = DiverseWasteDataset(repeats_per_image=40)
    print(f"Total augmented training samples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Initializing MobileNetV2 backbone...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Train top layers and classifier head
    for param in model.features[:-6].parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(model.last_channel, 128),
        nn.ReLU(),
        nn.Linear(128, len(CLASSES))
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001
    )

    epochs = 8
    print(f"Training for {epochs} epochs on CPU...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / total) * 100
        avg_loss = total_loss / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - Accuracy: {acc:.1f}%")

    # Save model weights
    save_path = MODELS_DIR / "waste_classifier.pth"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX
    }, save_path)
    print(f"Model saved successfully to: {save_path}")


if __name__ == "__main__":
    train_waste_model()
