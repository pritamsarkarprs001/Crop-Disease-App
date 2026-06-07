# 🌿 Agri-Scan AI: Crop Disease Predictor

This repository contains a complete Machine Learning pipeline and web application that detects 22 different crop diseases from images of plant leaves. 

### 🚀 Project Overview
* **Model:** EfficientNet-B0 (Trained via PyTorch on Kaggle)
* **Optimization:** Exported to ONNX format for low-latency CPU inference.
* **Image Processing:** Utilizes OpenCV with CLAHE (Contrast Limited Adaptive Histogram Equalization) to handle varied real-world lighting conditions.
* **Interface:** Built with Streamlit.

### 🛠️ How to Run Locally
To run this project on your own machine, follow these steps:

1. Clone this repository to your computer.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt


 **Dataset Link: https://www.kaggle.com/datasets/nirmalsankalana/crop-pest-and-disease-detection**