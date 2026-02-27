import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed"
MODEL_PATH = REPO_ROOT / "anesthesia_vit_full_model.pth"
BATCH_SIZE = 32


def evaluate():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"🔍 Evaluating on: {device_name}")

    # 1. Prepare Data (Same transforms as training)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=str(DATA_PATH), transform=transform)
    class_names = full_dataset.classes

    # Re-create the split to get the exact same test set
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    num_workers = 4 if use_cuda else 0
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
    )

    # 2. Load the Saved Model
    print(f"📥 Loading model from {MODEL_PATH}...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(class_names),
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
    )

    # Load the weights you just trained
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Get Predictions
    all_preds = []
    all_labels = []

    print("⚡ Running predictions on Test Set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Metrics
    print("\n📈 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("✅ Confusion matrix saved as confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
