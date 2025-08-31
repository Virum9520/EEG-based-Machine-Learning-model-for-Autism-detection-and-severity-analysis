# 🧠 EEG-Based Autism Spectrum Disorder (ASD) Detection and Severity Classification

This repository contains the implementation of a two-stage deep learning system for **EEG-based ASD detection and severity classification**.  
The project leverages **brain functional connectivity (BFC)** features derived from EEG signals, combined with **CNN–LSTM architectures**, to achieve high-accuracy classification.

---

## 📑 Table of Contents
- [Overview](#overview)
- [Methodology](#methodology)
- [Pipeline](#pipeline)
- [Visualizations](#visualizations)
  - [EEG Preprocessing](#1-eeg-preprocessing)
  - [Functional Connectivity](#2-functional-connectivity)
  - [Grayscale Image Construction](#3-grayscale-image-construction)
  - [Model Architecture](#4-model-architecture)
  - [Training Performance](#5-training-performance)
  - [Classification Results](#6-classification-results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results](#results)
- [Citation](#citation)

---

## 🔍 Overview
- **Stage 1**: Binary classification of ASD vs. Typically Developing (TD) individuals.  
- **Stage 2**: Severity classification of ASD into sub-groups based on EEG patterns.  
- **Key Features**: EEG preprocessing, brain functional connectivity (Pearson correlation), grayscale image representation, CNN–LSTM deep learning, hyperparameter tuning, and statistical validation.

---

## ⚙️ Methodology
1. **EEG Preprocessing**  
   - Artifact removal, filtering, and segmentation.  
2. **Functional Connectivity Analysis**  
   - Pearson correlation of EEG channel pairs.  
3. **Feature Transformation**  
   - Normalized BFC matrices → converted to grayscale images.  
4. **Deep Learning Model**  
   - CNN–LSTM hybrid network trained on connectivity images.  
5. **Evaluation**  
   - Accuracy, Precision, Recall, F1-score, ROC-AUC.  

---

## 🔗 Pipeline
![Pipeline](path/to/pipeline_diagram.png)

---

## 📊 Visualizations

### 1. EEG Preprocessing
- Raw EEG vs. filtered EEG signal  
![EEG Preprocessing](path/to/eeg_preprocessing.png)

---

### 2. Functional Connectivity
- Brain Functional Connectivity (BFC) matrices (ASD vs. TD)  
![Functional Connectivity](path/to/bfc_matrix.png)

---

### 3. Grayscale Image Construction
- Normalized connectivity matrix → grayscale representation  
![Grayscale Images](path/to/grayscale_images.png)

---

### 4. Model Architecture
- CNN–LSTM hybrid model for classification  
![CNN-LSTM Architecture](path/to/model_architecture.png)

---

### 5. Training Performance
- Loss and accuracy curves during training  
![Training Curves](path/to/training_curves.png)

---

### 6. Classification Results
- Confusion Matrix (Stage 1: ASD vs. TD)  
![Confusion Matrix](path/to/confusion_matrix.png)

- ROC Curve & AUC  
![ROC Curve](path/to/roc_curve.png)

- Severity Classification Results (Stage 2)  
![Severity Classification](path/to/severity_classification.png)

---

## 📦 Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
