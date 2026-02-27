import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTForImageClassification

# --- CONFIGURATION ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed"
KFOLD_RESULTS_PATH = REPO_ROOT / "kfold_results.json"
BATCH_SIZE = 32
EPOCHS = 5  # 5 epochs per fold for speed
LEARNING_RATE = 2e-5
K_FOLDS = 5


def kfold_validation():
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

    # --- K-FOLD SETUP ---
    indices = np.arange(len(full_dataset))
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    fold_accuracies = []

    print(f"\n{'=' * 60}")
    print(f"🧪 {K_FOLDS}-FOLD CROSS-VALIDATION (Vision Transformer)")
    print(f"{'=' * 60}\n")

    num_workers = 4 if use_cuda else 0
    pin_memory = use_cuda

    for fold, (train_idx, test_idx) in enumerate(kfold.split(indices), 1):
        print(f"\n{'=' * 60}")
        print(f"FOLD {fold}/{K_FOLDS}")
        print(f"{'=' * 60}\n")

        train_subset = Subset(full_dataset, train_idx)
        test_subset = Subset(full_dataset, test_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Initialize ViT model for each fold
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
            id2label={str(i): c for i, c in enumerate(class_names)},
            label2id={c: str(i) for i, c in enumerate(class_names)},
        )
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0

        # Training loop for each fold
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            loop = tqdm(train_loader, desc=f"Fold {fold} Epoch [{epoch+1}/{EPOCHS}]")
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

            # Validation after each epoch
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
            print(f"📊 Fold {fold} Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

            if accuracy > best_acc:
                best_acc = accuracy

        fold_accuracies.append(best_acc)
        fold_results.append({"fold": fold, "best_accuracy": best_acc})
        print(f"⭐ Fold {fold} Best Accuracy: {best_acc:.2f}%")

    # --- SUMMARY ---
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"\n✅ Cross-validation complete. Average Accuracy: {avg_accuracy:.2f}%")

    results = {
        "fold_results": fold_results,
        "average_accuracy": avg_accuracy,
        "num_folds": K_FOLDS,
        "epochs_per_fold": EPOCHS,
    }

    with open(KFOLD_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"💾 Results saved to {KFOLD_RESULTS_PATH}")


if __name__ == "__main__":
    kfold_validation()
