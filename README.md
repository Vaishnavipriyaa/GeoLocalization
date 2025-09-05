# 📌 Satellite Image Prediction Suite  

This repository contains **three deep learning solutions** designed for different satellite image prediction tasks:  

---

## 🔹 1. Angle Prediction (0–360°)  
- **Model:** Vision Transformer (`vit_large_patch16_224`)  
- **Approach:** Represented angles as **2D vectors (cos θ, sin θ)** to avoid wrapping issues near 0°/360°.  
- **Loss:** Cosine similarity between predicted and ground truth vectors.  
- **Optimizer:** AdamW + Cosine Annealing LR.  
- **Metric:** Mean Angular Absolute Error (MAAE).  
- **Highlight:** Robust and accurate angle prediction with **low angular error**.  
- **Files:** `main.ipynb`, `solution.csv`, [Model Weights](https://drive.google.com/file/d/1Ih4qc1lBmyxRgUaVMCTJlXHqMzP5T-M7/view?usp=sharing)  

---

## 🔹 2. Latitude & Longitude Regression  
- **Model:** EfficientNet-B0 (two separate heads for latitude & longitude).  
- **Approach:**  
  - Anomaly filtering to remove outlier coordinates.  
  - Min-max normalization for consistent training.  
  - Data augmentation with rotation & flipping.  
- **Loss:** Mean Squared Error (MSE).  
- **Optimizer:** Adam with learning rate decay (factor 0.5 every 5 epochs).  
- **Metric:** Validation MSE.  
- **Highlight:** Achieved **precise geographic coordinate predictions (<0.01 MSE)**.  
- **Files:** `main.ipynb`, `solution.csv`,  
  [Latitude Weights](https://drive.google.com/file/d/1FVue2lWwPVw-sEb45hAl0XscJrUx6HJJ/view?usp=sharing),  
  [Longitude Weights](https://drive.google.com/file/d/1WnQ-XiXKPSfze3QTOfkX4viC2HbdJ72L/view?usp=sharing)  

---

## 🔹 3. Region ID Classification (1–15)  
- **Model:** Swin Transformer (`swin_base_patch4_window7_224`)  
- **Approach:**  
  - **Loss:** Label Smoothing Cross-Entropy.  
  - **Optimizer:** AdamW + Cosine Annealing LR.  
  - **Inference:** Test Time Augmentation (TTA with 8 geometric transforms).  
- **Metric:** Classification Accuracy.  
- **Highlight:** Achieved **>94% accuracy** in region classification with TTA.  
- **Files:** `main.ipynb`, `solution.csv`, [Model Weights](https://drive.google.com/file/d/1IPxw9RmKMJRXS499Pki9xRUBFFz_sHGz/view?usp=sharing)  

---

## ✨ Key Features Across All Tasks
- Leveraged **transfer learning** with ViT, EfficientNet, and Swin Transformer.  
- Applied **advanced augmentations**, normalization, and early stopping.  
- Used **task-specific loss functions** (cosine similarity, MSE, label smoothing).  
- Best-performing models and predictions saved for reproducibility.  

---
