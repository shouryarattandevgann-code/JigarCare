# 🩺 JIGAR CARE: AI-Assisted Liver Fibrosis Staging & Reporting Studio

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.0-38bdf8.svg)](https://tailwindcss.com/)
[![Clinical Staging](https://img.shields.io/badge/METAVIR-F0--F4-success.svg)](#metavir-clinical-staging-reference)

**JIGAR CARE** is an end-to-end medical artificial intelligence system designed for accurate, non-invasive staging of chronic liver fibrosis from B-mode ultrasound imaging. 

Built with a fine-tuned **ResNet-50** deep convolutional neural network, the system automates **METAVIR classification (F0 to F4)**, provides visual explainability through **native PyTorch Grad-CAM activation mapping**, and generates **clinical-grade medical diagnostic PDF reports** for patients and healthcare providers.

---

## 🌟 Key Features

- **Modern Clinical Web Studio**: Responsive dark-mode clinical dashboard built with modern glassmorphism, responsive grid layouts, and zero external frontend build steps.
- **High-Performance Inference**: Sub-second inference latency (~400 ms) running seamlessly on CPU or CUDA-enabled GPUs.
- **Explainable AI (Native Grad-CAM)**: Backpropagates classification gradients directly to ResNet-50's final convolutional layer (`layer4[-1]`) to project acoustic tissue activation heatmaps. Includes an interactive opacity slider to blend raw scans with the Grad-CAM heatmap.
- **1-Click Preset Samples**: Instant test buttons for **`F0 (Normal)`**, **`F1 (Mild)`**, **`F2 (Moderate)`**, **`F3 (Severe)`**, and **`F4 (Cirrhosis)`** for one-click demonstration without needing to upload files manually.
- **Automated Medical PDF Reports**: Pure-Python medical report compilation (`fpdf2`) with patient demographics, AI classification metrics, dual ultrasound & Grad-CAM figures, radiological criteria, clinical recommendations, and physician signature block.
- **Production-Ready REST API**: Modular FastAPI backend with endpoints for health checks, batch samples, real-time predictions, and dynamic report downloads.

---

## 📋 METAVIR Clinical Staging Reference

| Stage | Classification | Histological / Anatomical Description | Key Ultrasound Findings | Clinical Management Plan |
| :---: | :--- | :--- | :--- | :--- |
| **F0** | **No Fibrosis** | Normal liver parenchyma without fibrous tissue expansion. | Smooth surface, homogeneous echotexture, normal vascular tapering. | Annual wellness assessment; lifestyle and metabolic risk screening. |
| **F1** | **Mild Fibrosis** | Fibrous expansion confined strictly to portal tracts without septa. | Minimal portal zone thickening; liver contour remains smooth. | Identify underlying etiology (NAFLD/viral/alcohol); monitor LFTs every 6–12 months. |
| **F2** | **Moderate Fibrosis** | Portal fibrosis with few fibrous septa extending into lobules. | Mild coarsened echotexture, slightly blunted hepatic edge. | Gastroenterology/hepatology consultation; confirm with FibroScan; initiate targeted therapy. |
| **F3** | **Severe Fibrosis** | Numerous bridging septa without established nodular cirrhosis. | Marked coarsening, undulating surface, blunted edges. | Comprehensive hepatology care; semiannual ultrasound & AFP surveillance; varices screening. |
| **F4** | **Cirrhosis** | Diffuse architectural distortion with regenerative nodules. | Coarse nodular surface, heterogeneous texture, portal hypertension signs. | Multidisciplinary cirrhosis management; 6-month HCC surveillance; EGD variceal prophylaxis. |

---

## 📁 Repository Structure

```text
Liver Fibrosis/
├── README.md                 # Project overview and quickstart guide
├── PROJECT_DOCUMENTATION.txt # Detailed clinical and technical documentation
├── requirements.txt          # Python dependencies
├── server.py                 # FastAPI backend server & REST API
├── check_dataset.py          # Quick dataset verification tool
├── eda.py                    # Exploratory Data Analysis script
├── .gitignore                # Git ignore rules for caches & virtualenvs
│
├── static/
│   └── index.html            # Modern Diagnostic Studio frontend (Tailwind + Lucide)
│
├── src/
│   ├── inference.py          # LiverFibrosisPredictor & native Grad-CAM engine
│   ├── report_generator.py   # Pure-Python medical PDF generator (fpdf2)
│   ├── preprocessing.py      # Duplicate removal & stratified 70/15/15 splitting
│   ├── train_resnet.py       # Fine-tuning training pipeline
│   └── evaluate_resnet.py    # Test set evaluation, metrics & confusion matrix
│
├── models/
│   ├── best_resnet50.pth     # Trained ResNet-50 PyTorch model checkpoint (~94 MB)
│   └── config.json           # Model configuration, normalization stats & class mapping
│
├── Dataset/                  # Raw ultrasound images categorized by METAVIR stage
│   ├── F0/                   # Normal scans
│   ├── F1/                   # Mild fibrosis scans
│   ├── F2/                   # Moderate fibrosis scans
│   ├── F3/                   # Severe fibrosis scans
│   └── F4/                   # Cirrhosis scans
│
└── results/                  # Generated plots, metrics, and data splits
    ├── eda/                  # Exploratory data analysis distributions & plots
    ├── preprocessing/        # Data split distributions & split CSV
    └── evaluation/           # Confusion matrices & evaluation reports
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- CUDA-compatible GPU (optional; CPU execution is fully supported)

### 2. Installation
Clone or navigate to the project directory and install the required dependencies:

```bash
# Optional: Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Web Diagnostic Studio
Start the FastAPI server:

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📡 REST API Reference

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Returns server status, device (`cpu` or `cuda`), and model information. |
| `GET` | `/api/samples` | Lists available preset sample ultrasound images for 1-click testing. |
| `GET` | `/api/sample/{stage}/{filename}` | Streams a specific preset ultrasound scan file. |
| `POST` | `/api/predict` | Accepts an ultrasound scan via multipart form data; returns predicted stage, confidence, probability distribution, and base64 Grad-CAM overlay. |
| `POST` | `/api/generate-report` | Accepts patient demographics and prediction data; streams an official medical PDF diagnostic report. |

---

## 🧠 Model Training & Evaluation

If you wish to re-train or evaluate the ResNet-50 model from scratch:

1. **Dataset Preprocessing & Stratified Splitting**:
   ```bash
   python src/preprocessing.py
   ```
   *Removes MD5 hash duplicates and stratifies images into a 70% train / 15% val / 15% test split.*

2. **Model Training**:
   ```bash
   python src/train_resnet.py --epochs_stage_a 5 --epochs_stage_b 10
   ```
   *Performs two-stage transfer learning (freezing feature backbone first, then fine-tuning high-level convolutional layers).*

3. **Evaluation on Test Set**:
   ```bash
   python src/evaluate_resnet.py
   ```
   *Evaluates performance, saves confusion matrices, and writes classification metrics to `results/evaluation/`.*

---

## ⚖️ Clinical Disclaimer

This software is an artificial intelligence diagnostic decision-support tool designed for investigational and educational purposes. Predictions and generated reports must be evaluated by licensed medical professionals (radiologists or hepatologists) in conjunction with clinical history, laboratory biomarkers, and physical examinations.

