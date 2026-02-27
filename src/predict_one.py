import glob
import sys
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification

# --- CONFIG ---
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "anesthesia_vit_full_model.pth"
DATA_GLOB = REPO_ROOT / "data" / "processed" / "*" / "*.png"
# Default sample image (can be overridden via CLI)
IMAGE_PATH = next(iter(glob.glob(str(DATA_GLOB))), None)


def predict_image(image_path):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load Model
    # Note: We hardcode classes here for simplicity since we aren't loading the dataset
    class_names = ["awake", "deep", "light"]

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(class_names),
        id2label={str(i): c for i, c in enumerate(class_names)},
        label2id={c: str(i) for i, c in enumerate(class_names)},
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Prepare Image
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    print(f"\nProcessing: {image_path}")
    print(f"Prediction: {class_names[predicted_class.item()].upper()}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")


if __name__ == "__main__":
    # Allow passing image path from command line
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = IMAGE_PATH
        if img_path is None:
            print("Could not find a sample image. Provide a path: python src/predict_one.py <image_path>")
            sys.exit(1)

    predict_image(img_path)
