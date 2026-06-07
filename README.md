# 🌿 Agri-Scan AI: Crop Disease Predictor

This repository contains a complete Machine Learning pipeline and web application that detects 22 different crop diseases from images of plant leaves. 

### 🚀 Project Overview
* **Model:** EfficientNet-B0 (Trained via PyTorch on Kaggle)
* **Optimization:** Exported to ONNX format for low-latency CPU inference.
* **Image Processing:** Utilizes OpenCV with CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle varied real-world lighting conditions.
* **Interface:** Built with Streamlit.

  
## 🚀 Key Features
* **Production-Grade Inference:** Uses an **EfficientNet-B0** backbone exported to **ONNX**, enabling sub-millisecond inference on low-power CPU hardware.
* **Robust Image Pipeline:** Employs **CLAHE (Contrast Limited Adaptive Histogram Equalization)** within the preprocessing pipeline to ensure the model remains accurate under inconsistent lighting conditions (e.g., cloudy, high-noon, or shadowed field images).
* **Anti-Overfitting Logic:** Trained with **Label Smoothing**, **AdamW weight decay**, and **Early Stopping** to ensure high generalization to unseen real-world data.
* **Intuitive Interface:** A clean, user-centric web interface built with **Streamlit** for instant analysis and confidence reporting.



## 🏗️ Architecture & Pipeline
The project follows a standard MLOps lifecycle:
1. **Sanitization:** Automated corruption filtering and structural integrity auditing.
2. **Augmentation:** Heavy geometric and chromatic augmentation to simulate agricultural environmental noise.
3. **Training:** Fine-tuned on the agricultural dataset with stratified validation.
4. **Serialization:** Graph translation to ONNX for seamless deployment outside of PyTorch environments.


## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, `timm` (PyTorch Image Models)
* **Optimization:** ONNX, ONNXRuntime
* **Deployment:** Streamlit, OpenCV
* **Environment:** Kaggle Kernels & VS Code

  
### 🛠️ How to Run Locally
To run this project on your own machine, follow these steps:

1. Clone this repository to your computer.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


 **Dataset Link: https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection**
