# Skin Lesion Classification with EfficientNet-B4

A deep learning model for classifying dermatoscopic images into **seven skin lesion types**, including malignant melanoma. Built in PyTorch on the HAM10000 dataset, the model reaches **84.63% accuracy** on an independent test set and is designed to act as an automated "second opinion" for early skin cancer screening.

> This repository covers the **AI / deep learning component** of a larger project — an *AI-Assisted Dermatoscope for Early Diagnosis of Skin Cancer* — developed at Middle East Technical University (METU). The optical hardware (dual-path hybrid optics, cross-polarized and multispectral illumination) was developed separately; this README focuses on the model that processes the captured images.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Training Setup](#training-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Clinical Interpretation](#clinical-interpretation)
- [Getting Started](#getting-started)
- [Authors](#authors)
- [References](#references)

---

## Overview

Malignant melanoma is an aggressive skin cancer where early detection is critical for survival, yet visual diagnosis is often subjective. Deep learning has shown dermatologist-level diagnostic potential, so this component builds a convolutional neural network that classifies a dermatoscopic image into one of seven lesion categories in real time.

The model is meant to **augment** clinical decision-making — providing an objective, automated screen that flags potentially malignant cases while the clinician retains final judgment.

---

## Dataset

- **Source:** [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
- **Size:** 10,015 images across 7 classes.

| Code | Diagnosis | Type |
|---|---|---|
| `nv` | Melanocytic nevi | Benign (common moles) |
| `mel` | **Melanoma** | **Malignant — most aggressive** |
| `bkl` | Benign keratosis | Benign (seborrheic keratoses, solar lentigines) |
| `bcc` | Basal cell carcinoma | Cancerous, low metastasis |
| `akiec` | Actinic keratoses | Pre-cancerous |
| `vasc` | Vascular lesions | Benign (blood vessel lesions) |
| `df` | Dermatofibroma | Benign (skin nodules) |

The dataset is **heavily imbalanced** — benign nevi (`nv`) dominate, while critical classes like melanoma are underrepresented. This is addressed during training (see below).

---

## Model

- **Architecture:** EfficientNet-B4
- **Framework:** PyTorch
- **Why EfficientNet-B4:** its compound-scaling design gives a strong accuracy-to-efficiency ratio, keeping the model lightweight enough for potential embedded / portable use in a handheld device.

---

## Training Setup

- **Class imbalance handling:** a **Weighted Random Sampler** oversamples minority classes (e.g. melanoma) during training, preventing the model from collapsing toward the dominant benign-nevi class.
- The model was trained to classify the seven lesion types and evaluated on a held-out independent test set (N = 1503).

> If you want, add your exact hyperparameters here — learning rate, optimizer, batch size, number of epochs, image resolution, and augmentations — so the run is reproducible.

---

## Evaluation Metrics

With an imbalanced medical dataset, accuracy alone is misleading, so the following metrics were prioritized:

- **Precision** — minimizes false positives, reducing unnecessary biopsies.
- **Recall (Sensitivity)** — the most important metric for melanoma, since it minimizes false negatives (missed cancers).
- **F1-Score** — the harmonic mean of precision and recall, giving a balanced view under class imbalance.

---

## Results

Evaluated on an independent test set (**N = 1503**). **Overall accuracy: 84.63%**.

| Class | Diagnosis | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| `akiec` | Actinic Keratoses | 0.77 | 0.84 | 0.80 | 49 |
| `bcc` | Basal Cell Carcinoma | 0.80 | 0.86 | 0.83 | 77 |
| `bkl` | Benign Keratosis | 0.73 | 0.72 | 0.73 | 165 |
| `df` | Dermatofibroma | 0.88 | 0.88 | 0.88 | 17 |
| `mel` | **Melanoma (Malignant)** | 0.57 | 0.71 | 0.63 | 167 |
| `nv` | Melanocytic Nevi | 0.94 | 0.89 | 0.91 | 1006 |
| `vasc` | Vascular Lesions | 0.83 | 0.91 | 0.87 | 22 |
| **AVG** | **Weighted Average** | **0.86** | **0.85** | **0.85** | **1503** |

---

## Clinical Interpretation

- **Melanoma sensitivity:** the model reached a recall of **0.71** for melanoma. The deliberate trade-off with lower precision (0.57) reflects a **safety-first** design — it is better to over-flag a suspicious lesion than to miss a malignant one.
- **Strong performance on common classes:** melanocytic nevi (`nv`) achieved an F1 of 0.91, meaning the model reliably handles the most frequent benign cases.
- **Role in the system:** the AI mitigates human subjectivity and reduces the risk of overlooking malignant cases, while the clinician verifies predictions through the device's optical modes.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Main libraries:** `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`.

Download the HAM10000 dataset from the [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and place it under `data/`.

```bash
# Example commands — adjust to your scripts
python src/train.py --epochs 30 --batch-size 32
python src/evaluate.py --checkpoint checkpoints/best.pth
```

---

## Authors

- **Buğra Akkuş** — Department of Statistics, METU *(AI / deep learning)*
- **Ahmet Sünbül** — Department of Physics, METU *(optical system)*

**Advisor:** Prof. Dr. Ahmet Bingül, Department of Physics, METU

*Supported by the AdımODTÜ Undergraduate Research Program and the TÜBİTAK 2209-A Research Project Support Programme for Undergraduate Students.*

---

## References

1. A. Esteva et al., *Dermatologist-level classification of skin cancer with deep neural networks*, Nature, 542(7639):115–118, 2017.
2. P. Tschandl et al., *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Sci. Data, 5:180161, 2018.
3. M. Tan, Q. Le, *EfficientNet: Rethinking model scaling for convolutional neural networks*, ICML, pp. 6105–6114, 2019.
4. J. P. Tomtishen et al., *Multi-class CNN for classification of multispectral and autofluorescence skin lesion clinical images*, J. Clin. Med., 11(10):2833, 2022.
5. V. I. Batshev et al., *Optical system of a compact dermatoscope with a videocapillaroscopy channel*, J. Opt. Technol., 90(11):679–683, 2023.
