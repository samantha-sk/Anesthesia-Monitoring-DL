# Anesthesia Depth Classification using Vision Transformer (ViT)

## 📋 Overview

This project implements a **Vision Transformer (ViT) based deep learning model** to classify anesthesia depth levels into three categories: **Awake**, **Deep**, and **Light**. The model is trained and evaluated on a dataset of 33,433 labeled images.

---

## 📂 Project Structure

```
anesthesia_vit_full_model/
├── data/
│   ├── processed/
│   │   ├── awake/        # Awake anesthesia depth images
│   │   ├── deep/         # Deep anesthesia depth images
│   │   └── light/        # Light anesthesia depth images
│   └── raw/              # Raw .mat files (24 cases)
│
├── src/
│   ├── train_vit.py              # Main training script
│   ├── evaluate_model.py          # Standard evaluation
│   ├── predict_one.py            # Single image prediction
│   ├── compare_models.py          # Baseline model comparison
│   ├── kfold_validation.py        # 5-Fold cross-validation
│   ├── evaluate_medical_metrics.py # Medical metrics (Sensitivity, Specificity, ROC-AUC)
│   └── preprocess.py             # Data preprocessing utilities
│
├── anesthesia_vit_full_model.pth  # Trained model weights
├── requirements.txt               # Python dependencies
├── confusion_matrix.png           # Evaluation visualization
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Libraries:**
- PyTorch 2.5.1 (with CUDA 12.1)
- Transformers (HuggingFace ViT)
- torchvision
- scikit-learn
- matplotlib, seaborn

### 2. Train the Model

```bash
python src/train_vit.py
```

**Training Configuration:**
- Model: ViT-Base (google/vit-base-patch16-224-in21k)
- Batch Size: 32
- Epochs: 10
- Learning Rate: 2e-5
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Data Split: 80/20 (Train/Test)

### 3. Evaluate the Model

```bash
python src/evaluate_model.py
```

Generates:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix (`confusion_matrix.png`)

### 4. Predict on Single Image

```bash
python src/predict_one.py
```
---

## 🔬 Advanced Evaluation

### Baseline Model Comparison

Compare ViT with ResNet-50 and EfficientNet-B0:

```bash
python src/compare_models.py
```

**Results stored in:** `model_comparison_results.json`

### 5-Fold Cross-Validation

Validate model robustness with k-fold cross-validation:

```bash
python src/kfold_validation.py
```

**Results stored in:** `kfold_results.json`

### Medical Metrics (Sensitivity, Specificity, ROC-AUC)

Evaluate model using medical domain standards:

```bash
python src/evaluate_medical_metrics.py
```

**Output:**
- Sensitivity & Specificity per class
- ROC-AUC curves (`roc_curves.png`)
- Confusion matrix
- Medical metrics JSON file

---

## 🧠 Model Architecture

**Vision Transformer (ViT-Base)**

- **Input:** 224×224 RGB images
- **Patch Size:** 16×16 (resulting in 196 patches)
- **Embedding Dimension:** 768
- **Transformer Blocks:** 12
- **Attention Heads:** 12
- **Parameters:** ~86M
- **Pre-trained on:** ImageNet-21K
- **Fine-tuned for:** 3-class anesthesia depth classification

**Key Innovation:** ViTs treat images as sequences of patches, enabling better long-range dependency learning compared to CNNs.

---

## 🔧 Configuration Details

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Resize to ViT input size
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize to ImageNet stats
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    )
])
```

### Training Hyperparameters
```python
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
DEVICE = "cuda"  # GPU acceleration
num_workers = 4  # Parallel data loading
pin_memory = True  # GPU memory optimization
```

---

## 🎯 Medical Application Notes

This model is designed to classify anesthesia depth, which is critical for:
- **Patient Safety:** Ensuring adequate anesthesia during surgery
- **Resource Optimization:** Preventing over/under-anesthesia
- **Real-time Monitoring:** Automated depth level detection

### Important Limitations
1. **Awake Class Low Recall (79%):** The model sometimes misclassifies Awake as Light. Consider:
   - Class weighting in loss function
   - Additional Awake-specific training examples
   - Ensemble methods with other models

2. **Medical Validation Required:** This is a research prototype. Clinical deployment requires:
   - FDA/regulatory approval
   - Clinical validation on diverse patient populations
   - Integration with anesthesia monitoring equipment

---

## 📈 Dataset Information

**Total Images:** 33,433  
**Cases:** 24 (from MATLAB .mat files)  
**Classes:** 3 (Awake, Deep, Light)

**Distribution:**
- Awake: ~3.4% (1,135 images)
- Deep: ~9.1% (3,051 images)
- Light: ~7.5% (2,501 images)

**Note:** Dataset is imbalanced, with Deep anesthesia overrepresented. This affects model learning and may explain Awake class lower recall.
