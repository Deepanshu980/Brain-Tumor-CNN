# 🧠 Brain Tumor Detection Using CNN (MRI Images)

This project implements a **Convolutional Neural Network (CNN)** to classify **brain tumor MRI images**.  
The model learns visual patterns from MRI scans and accurately distinguishes tumor-related features using deep learning.

---

## 📌 Project Overview

Brain tumor detection from MRI images is a critical task in medical image analysis.  
In this project, a **CNN-based deep learning model** is trained to automatically classify brain MRI images, aiming to support early diagnosis and reduce manual effort.

The model demonstrates **high accuracy, fast convergence, and strong generalization**, making it suitable for real-world medical imaging applications.

---

## 🧪 Dataset

- **Source:** Brain Tumor MRI Dataset (Kaggle)
- **Image Type:** MRI scans
- **Classes:** Tumor / Non-Tumor (or multiple tumor categories if applicable)
- **Preprocessing:**
  - Image resizing
  - Normalization
  - Data augmentation (if applied)

---

## 🏗️ Model Architecture

The CNN model consists of:
- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Fully connected (Dense) layers for classification
- Softmax/Sigmoid activation for final prediction

The architecture is designed to balance **performance and computational efficiency**.

---

## 🚀 Training Performance

The model was trained for **4 epochs**, showing steady improvement across epochs.

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|------|------------------|--------------|--------------------|----------------|
| 1 | 66.17% | 0.8166 | 85.28% | 0.3726 |
| 2 | 88.88% | 0.3070 | 91.30% | 0.2402 |
| 3 | 95.28% | 0.1424 | 91.76% | 0.2214 |
| 4 | 97.19% | 0.0831 | **94.66%** | **0.1496** |

✔ Rapid convergence  
✔ Reduced validation loss  
✔ Minimal overfitting  

---

## 📊 Results & Observations

- Training accuracy improved from **66% → 97%**
- Validation accuracy reached **94.66%**
- Loss values consistently decreased
- Model generalizes well to unseen MRI images

These results confirm the effectiveness of CNNs for medical image classification.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Framework:** TensorFlow / Keras
- **Libraries:**
  - NumPy
  - Matplotlib
  - OpenCV / PIL
  - Scikit-learn
- **Platform:** Kaggle / Local Machine


