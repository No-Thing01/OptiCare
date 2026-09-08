# OptiCare 👁️🤖
**AI-Powered Diabetic Retinopathy Diagnostic Platform**

OptiCare is a high-precision, hybrid-ensemble machine learning platform built to detect and grade Diabetic Retinopathy (DR) from retinal fundus images. It is engineered to overcome real-world clinical challenges such as variable camera lighting and class imbalance, specifically tailored for Indian patient demographics.

## 🏆 The "Hybrid Ensemble" Architecture
Our backend does not rely on a single neural network. We engineered a dual-system approach that acts as a clinical fail-safe:
1. **PyTorch ResNet-50 (Deep Learning):** A customized Convolutional Neural Network trained to classify retinal scans into 5 severity levels (0-4).
2. **MATLAB Core (Mathematical Safety Net):** While PyTorch handles semantic feature extraction, our backend simultaneously runs a MATLAB script that performs Hessian-based matrix lesion counting to physically count microaneurysms. 
*If the PyTorch AI underestimates the severity of a scan, the MATLAB system mathematically overrides it to protect the patient.*

## 📊 Dataset Strategy: Preventing Domain Bias
Most DR classification models fail in the real world because they are trained on a single noisy dataset (like APTOS) and become biased to specific fundus cameras. OptiCare solves this:
*   **Base Training:** We aggregated over 4,500 images from **APTOS 2019** and **Messidor-2** to teach the model general DR features.
*   **CLAHE Pipeline Alignment:** We eliminated "Domain Shift" by running Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L*a*b* color space dynamically during the PyTorch training loop, ensuring the AI trains on the exact same high-contrast images it sees in production.
*   **SIH Specialization:** We utilize the **IDRiD (Indian Diabetic Retinopathy Image Dataset)** to fine-tune and benchmark the model specifically for the Indian demographic.

## 🚀 Live Demo & Deployment
Our frontend is fully responsive (Desktop/Tablet/Mobile) and hosted entirely on GitHub Pages. The backend is built on Flask and funneled securely to the frontend via Ngrok tunnels.

**To run the system locally:**
```powershell
# 1. Start the Flask Backend (Make sure you are in your .venv!)
python backend/app.py

# 2. Expose the port using Ngrok
ngrok http 5000
```
*(Copy the Ngrok URL into `frontend/script.js`, push to GitHub, and your live site is instantly connected!)*

## 🧠 Explainable AI (XAI)
To build trust with doctors, OptiCare does not operate as a "black box". Every diagnosis generates:
*   A **Semantic Heatmap** (Grad-CAM) showing exactly which blood vessels or lesions the PyTorch model focused on.
*   A **3D Morphological Topography Map** rendering the retina surface.
*   An auto-generated, text-based **Clinical Report** with a confidence score and action plan.
