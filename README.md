# 🫁 Chest X-Ray Multi-Disease Classifier

> DenseNet-121 trained on NIH CXR8 for simultaneous detection of 14 thoracic pathologies — approaching CheXNet (Stanford) benchmark performance.

[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/Rugved2607/cxr-disease-classifier)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📊 Results

| Metric | This Model | CheXNet (Stanford) |
|---|---|---|
| **Mean AUC-ROC** | **0.8309** | 0.8414 |
| Effusion | **0.9040** | 0.8638 ✅ |
| Edema | **0.9182** | 0.8878 ✅ |
| Emphysema | **0.8900** | 0.8371 ✅ |
| Cardiomegaly | **0.8834** | 0.8926 |
| Pneumothorax | **0.8585** | 0.8887 |
| Hernia | **0.9162** | 0.9164 |
| Fibrosis | **0.7897** | 0.8047 |
| Atelectasis | **0.8168** | 0.8094 ✅ |
| Consolidation | **0.8111** | 0.7901 ✅ |
| Pleural Thickening | **0.8072** | 0.8062 ✅ |

> Exceeds CheXNet on **6 out of 14** pathologies in 5 training epochs.

---

## 🏗️ Architecture

```
Input (224×224×3)
        ↓
DenseNet-121 Backbone (pretrained ImageNet)
— 4 Dense Blocks with skip connections
— Feature reuse preserves fine-grained pathology details
        ↓
Global Average Pooling (1024 features)
        ↓
Linear (1024 → 15)
        ↓
Sigmoid → 15 independent disease probabilities
```

**Why DenseNet-121?**
Dense connections preserve subtle texture features critical for pathology detection. Validated in the original CheXNet paper on this exact dataset. Achieves strong performance with only 7.4M parameters vs ResNet50's 24.5M.

---

## 📁 Dataset

**NIH ChestX-ray14** — [Download here](https://nihcc.app.box.com/v/ChestXray-NIHCC)

| Split | Images | Patients |
|---|---|---|
| Train | 77,988 | ~25,200 |
| Validation | 8,536 | ~2,800 |
| Test | 25,596 | 389 |
| **Total** | **112,120** | **30,805** |

**14 Pathologies:** Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia

**Key challenge:** Severe class imbalance — No Finding (60,361 images) vs Hernia (227 images). Handled via BCEWithLogitsLoss without pos_weight after empirical validation.

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Rugved2607/CXR-Disease-Classification.git
cd CXR-Disease-Classification

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```

---

## 📦 Requirements

```
torch
torchvision
gradio
grad-cam
Pillow
numpy
scikit-learn
```

---

## 🔥 Features

- **Multi-label classification** — detects multiple diseases simultaneously in one X-ray
- **Grad-CAM visualization** — heatmap per detected disease showing model attention regions
- **Gradio UI** — upload any chest X-ray, get predictions + heatmaps instantly
- **Live demo** — deployed on HuggingFace Spaces, no setup required

---

## 🖥️ Demo

Try it live → [huggingface.co/spaces/Rugved2607/cxr-disease-classifier](https://huggingface.co/spaces/Rugved2607/cxr-disease-classifier)

---

## 🧪 Training Details

| Parameter | Value |
|---|---|
| Architecture | DenseNet-121 |
| Pretrained | ImageNet |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 5 |
| Loss | BCEWithLogitsLoss |
| Image Size | 224×224 |
| Augmentation | Random horizontal flip |

---

## 📈 Training Curve

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 0.1756 | 0.1638 |
| 2 | 0.1612 | 0.1604 |
| 3 | 0.1563 | 0.1578 |
| 4 | 0.1521 | 0.1589 |
| 5 | 0.1471 | 0.1593 |

---

## 📄 References

- [CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning](https://arxiv.org/abs/1711.05225) — Rajpurkar et al., Stanford, 2017
- [NIH ChestX-ray14 Dataset](https://arxiv.org/abs/1705.02315) — Wang et al., NIH, 2017
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Huang et al., 2017

---

## ⚠️ Disclaimer

This tool is for **research purposes only**. Not intended for clinical diagnosis or medical decision-making. Always consult a qualified radiologist for medical interpretation.

---

## 👤 Author

**Rugved Deshpande**  
[![GitHub](https://img.shields.io/badge/GitHub-Rugved2607-black)](https://github.com/Rugved2607)
[![HuggingFace](https://img.shields.io/badge/🤗-Rugved2607-yellow)](https://huggingface.co/Rugved2607)
