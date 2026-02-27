import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm
from transformers import ViTForImageClassification

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed"
RESULTS_PATH = REPO_ROOT / "model_comparison_results.json"
BATCH_SIZE = 32
EPOCHS = 3  # Reduced to 3 for faster comparison
LEARNING_RATE = 2e-5


def train_and_evaluate_model(model, train_loader, test_loader, device, model_name, epochs=EPOCHS, use_cuda=False):
    """Train and evaluate a model."""
    print(f"\n{'=' * 60}")
    print(f"🚀 Training {model_name}...")
    print(f"{'=' * 60}\n")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    training_history = {"epochs": [], "train_loss": [], "test_acc": []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"{model_name} Epoch [{epoch+1}/{epochs}]")
        for images, labels in loop:
            images = images.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            outputs = model(images)
            # Handle both direct logits and ViT output format
            if hasattr(outputs, "logits"):
                loss = criterion(outputs.logits, labels)
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=use_cuda)
                labels = labels.to(device, non_blocking=use_cuda)

                outputs = model(images)
                if hasattr(outputs, "logits"):
                    _, predicted = torch.max(outputs.logits, 1)
                else:
                    _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(running_loss / len(train_loader))
        training_history["test_acc"].append(accuracy)

        print(f"📊 {model_name} Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            print(f"⭐ New Best {model_name}! ({accuracy:.2f}%)")

    return best_acc, training_history


def compare_models():
    # --- DEVICE CHECK ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"✅ Using device: {device_name}\n")

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

    # 80/20 Train-Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    results = {}

    # --- 1. RESNET-50 ---
    print("\n" + "=" * 60)
    print("MODEL 1: ResNet-50")
    print("=" * 60)

    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

    resnet_acc, resnet_history = train_and_evaluate_model(
        resnet, train_loader, test_loader, device, "ResNet-50", use_cuda=use_cuda
    )
    results["ResNet-50"] = {"best_accuracy": resnet_acc, "history": resnet_history}

    # --- 2. EFFICIENTNET-B0 ---
    print("\n" + "=" * 60)
    print("MODEL 2: EfficientNet-B0")
    print("=" * 60)

    efficientnet = models.efficientnet_b0(pretrained=True)
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

    efficient_acc, efficient_history = train_and_evaluate_model(
        efficientnet, train_loader, test_loader, device, "EfficientNet-B0", use_cuda=use_cuda
    )
    results["EfficientNet-B0"] = {"best_accuracy": efficient_acc, "history": efficient_history}

    # --- 3. VISION TRANSFORMER (ViT) ---
    print("\n" + "=" * 60)
    print("MODEL 3: Vision Transformer (ViT-Base)")
    print("=" * 60)

    vit = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
    )

    vit_acc, vit_history = train_and_evaluate_model(
        vit, train_loader, test_loader, device, "Vision Transformer", use_cuda=use_cuda
    )
    results["Vision-Transformer"] = {"best_accuracy": vit_acc, "history": vit_history}

    # --- SAVE RESULTS ---
    print("\n" + "=" * 60)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1]["best_accuracy"], reverse=True)

    for rank, (model_name, data) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name:25} -> {data['best_accuracy']:.2f}% Accuracy")

    # Save to JSON
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    compare_models()
