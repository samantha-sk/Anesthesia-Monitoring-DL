"""
SOLUTION: Addressing Low Awake Class Recall (79%)

Problem: The model has only 79% recall for the Awake class, meaning it misses
some awake cases. In medical applications, missing an awake patient is critical.

Solutions Implemented:
1. Class-weighted loss function
2. Over-sampling the Awake class
3. Focal loss for harder examples
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTForImageClassification

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed"
MODEL_SAVE_PATH = REPO_ROOT / "anesthesia_vit_improved_model.pth"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5


class FocalLoss(nn.Module):
    """Focal Loss - emphasizes hard examples."""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction="none")(inputs, targets)
        p = torch.exp(-ce_loss)
        loss = (1 - p) ** self.gamma * ce_loss
        return loss.mean()


def train_improved_model():
    # --- DEVICE CHECK ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print("=" * 70)
    print(f"✅ Using device: {device_name}")
    print("=" * 70)
    print()

    # --- LOAD DATA ---
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    if not DATA_PATH.exists():
        print(f"❌ Error: Data folder not found at {DATA_PATH}")
        return

    full_dataset = datasets.ImageFolder(root=str(DATA_PATH), transform=transform)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"📂 Found {len(full_dataset)} images. Classes: {class_names}\n")

    # Compute class weights for imbalance
    class_counts = np.bincount(full_dataset.targets)
    class_weights = 1.0 / class_counts
    samples_weight = class_weights[full_dataset.targets]
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    # Train/Test split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Initialize ViT model
    print("🤖 Initializing ViT-Base (HuggingFace)...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
    )
    model.to(device)

    # Loss and optimizer
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    print("🚀 STARTING TRAINING (Improved)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            images = images.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)

                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"📊 Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ New Best Model Saved! ({accuracy:.2f}%)")

    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train_improved_model()
