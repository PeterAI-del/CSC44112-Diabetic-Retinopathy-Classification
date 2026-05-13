# CSC-44112: Diabetic Retinopathy Grading
## EfficientNet-B4 vs Swin-Tiny Vision Transformer

**Module:** Advanced Applications of AI and Machine Learning (CSC-44112)  
**Student ID:** 25021315  
**Programme:** MSc Artificial Intelligence and Data Science  
**University:** Keele University  
**Academic Year:** 2025–2026  

---

## Project Overview
This project implements and compares two deep learning architectures
for five-class diabetic retinopathy severity grading on the APTOS
2019 retinal fundus image dataset.

**Research Question:** Do Vision Transformers outperform CNNs for
ordinal DR grading under constrained training conditions?

---

## Key Results

| Model | Loss Function | Test QWK | Test Accuracy |
|---|---|---|---|
| EfficientNet-B4 | Focal Loss | 0.772 | 64.7% |
| EfficientNet-B4 | Weighted CE | 0.824* | — |
| **Swin-Tiny** | **Focal Loss** | **0.902** | **81.1%** |

*Validation QWK only

---

## Dataset
APTOS 2019 Blindness Detection (pre-resized 224×224)  
Source: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data  
Classes: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative)  
Total images: 3,662 | Train: 2,563 | Val: 549 | Test: 550

---

## Repository Structure
## Repository Structure
```text
CSC44112-Diabetic-Retinopathy-Classification/
│
├── APTOS_DR_Classification.ipynb   ← Main notebook (all sections)
│
├── figures/                         ← All generated figures
│   ├── fig1_class_distribution.png
│   ├── fig2_sample_images.png
│   ├── fig3_clahe_comparison.png
│   ├── fig4_pixel_intensity.png
│   ├── fig5_data_splits.png
│   ├── fig_efficientnet_curves.png
│   ├── fig_swin_curves.png
│   ├── fig_confusion_matrices.png
│   ├── fig_auc_curves.png
│   ├── fig_calibration.png
│   └── fig_gradcam.png
│
└── README.md
```
---

## Notebook Sections
| Section | Report Section | Description |
|---|---|---|
| Cells 1–6 | Setup | Install, auth, dataset, constants |
| Cells 7–12 | Section 3: EDA | Class distribution, CLAHE, splits |
| Cells 13–23 | Section 4: Methodology | Models, training, ablation |
| Cells 24–31 | Section 5: Results | Metrics, figures, Grad-CAM |

---

## How to Run
1. Open `APTOS_DR_Classification.ipynb` in Google Colab
2. Add Kaggle credentials to Colab Secrets:
   - `KAGGLE_USERNAME` and `KAGGLE_KEY`
3. Run all cells from top to bottom
4. All figures save automatically to `figures/`

---

## Requirements
torch, timm==0.9.12, opencv-python-headless,
albumentations, scikit-learn, matplotlib,
seaborn, grad-cam, thop, pandas, numpy

---

## References
- Gulshan et al. (2016) JAMA — DR deep learning baseline
- Liu et al. (2021) ICCV — Swin Transformer
- Tan and Le (2019) ICML — EfficientNet
- Full reference list in report PDF
