import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    hamming_loss,
    roc_auc_score,
    roc_curve,
    sensitivity_specificity_support,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import ViTForImageClassification

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed"
MODEL_PATH = REPO_ROOT / "anesthesia_vit_full_model.pth"
BATCH_SIZE = 32


def evaluate_with_medical_metrics():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"🔍 Evaluating on: {device_name}")

    # 1. Prepare Data
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    full_dataset = datasets.ImageFolder(root=str(DATA_PATH), transform=transform)
    class_names = full_dataset.classes
    num_classes = len(class_names)

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
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
    )

    # Load the weights you just trained
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Get Predictions
    print("⚡ Running predictions on Test Set...")

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=use_cuda)
            labels = labels.to(device, non_blocking=use_cuda)

            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            _, predicted = torch.max(outputs.logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 4. CLASSIFICATION REPORT
    print("\n" + "=" * 60)
    print("📈 Classification Report:")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. CONFUSION MATRIX
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

    # 6. ROC-AUC (One-vs-Rest)
    labels_binarized = label_binarize(all_labels, classes=range(num_classes))
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_binarized.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], label=f"{class_name} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    print("✅ ROC curves saved as roc_curves.png")

    # 7. MEDICAL METRICS
    sensitivity, specificity, support = sensitivity_specificity_support(
        all_labels, all_preds, average=None
    )
    hamming = hamming_loss(all_labels, all_preds)
    macro_auc = roc_auc_score(labels_binarized, all_probs, average="macro")

    print("\n" + "=" * 60)
    print("🩺 Medical Metrics")
    print("=" * 60)
    for i, cls in enumerate(class_names):
        print(
            f"{cls:<10} | Sensitivity: {sensitivity[i]:.3f} | "
            f"Specificity: {specificity[i]:.3f}"
        )
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")


if __name__ == "__main__":
    evaluate_with_medical_metrics()
