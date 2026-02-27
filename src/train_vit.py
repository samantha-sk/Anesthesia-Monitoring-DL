import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from transformers import ViTForImageClassification
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
DATA_PATH = r"D:\8th_sem_project\data\processed"
MODEL_SAVE_PATH = "anesthesia_vit_full_model.pth"
BATCH_SIZE = 32         
EPOCHS = 10             
LEARNING_RATE = 2e-5    

def train_model():
    # --- 1. STRICT GPU CHECK ---
    print("------------------------------------------------")
    print("🔍 CHECKING GPU STATUS...")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ SUCCESS: Found GPU -> {gpu_name}")
    else:
        print("⚠️ CUDA not available; falling back to CPU.")
    print("------------------------------------------------\n")
    device = torch.device("cuda" if use_cuda else "cpu")


    # 2. DATA AUGMENTATION & TRANSFORMS
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load Dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data folder not found at {DATA_PATH}")
        return

    full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    print(f"📂 Found {len(full_dataset)} images. Classes: {class_names}")

    # 80/20 Train-Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoader 
    # NOTE: If you get "BrokenPipeError", set num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 3. INITIALIZE ViT MODEL
    print("🤖 Initializing ViT-Base...")
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224-in21k',
        num_labels=num_classes,
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)}
    )
    model.to(device) # Move model to GPU

    # Loss and Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 4. TRAINING LOOP
    best_acc = 0.0
    
    print("🚀 STARTING TRAINING...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            # FORCE DATA TO GPU
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs.logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 5. VALIDATION AFTER EACH EPOCH
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"📊 Epoch {epoch+1} Accuracy: {accuracy:.2f}%")

        # Save the best model automatically
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"⭐ New Best Model Saved! ({accuracy:.2f}%)")

    print(f"\n✅ Training Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()