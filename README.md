# 🧠 Handwritten Digit Recognition System

Welcome to my Machine Learning project! This system is built to recognize handwritten digits (0–9) using a neural network trained on the MNIST dataset. It's a beginner-friendly dive into image classification, powered by Python and deep learning.

---

## ✨ Features

- Recognizes digits from handwritten images with high accuracy 🧮
- Built using **TensorFlow/Keras**
- Trained on the **MNIST** dataset (60,000+ images)
- Supports model evaluation and prediction
- Visualizes results for easy interpretation

---

## 🔧 Tech Stack

- Python 🐍
- TensorFlow / Keras
- NumPy & Matplotlib
- Jupyter Notebook

---

## 🚀 How It Works

1. Load and preprocess the MNIST dataset
2. Define and train a neural network model
3. Evaluate performance on test data
4. Visualize predictions and accuracy

---

## 📂 Project Structure

DigitRecognitionWebApp/ 
├── data/ # Dataset (auto-loaded from Keras) 
├── digits
    ├── digit1.png
    ├── digit2.png
    ├── digit3.png
    ├── digit4.png
    ├── digit5.png
    ├── digit6.png
    ├── digit7.png
    ├── digit8.png
    ├── digit9.png
    ├── digit10.png
    ├── testing.png
    ├── usoh5.png
├── static/uploads
    ├── digit1.png
    ├── digit2.png
    ├── digit3.png
    ├── digit4.png
    ├── digit5.png
    ├── digit6.png
    ├── digit7.png
    ├── digit8.png
    ├── digit9.png
    ├── digit10.png
    ├── testing.png
    ├── usoh5.png
├── templates
    ├── index.html
├── app.py # Script for training and prediction 
├── handwrittenDigit_ann.keras/ # Saved trained model 
├── hdr.ipynb/ # Jupyter Notebooks (exploration, training) 
├── README.md # You are here!

---

## 🧪 Getting Started

Clone the repo and install dependencies:

```bash
git clone https://github.com/LerieV/handwrittenDigitRecognitionSystem.git
cd DigitRecognitionWebApp
pip install -r requirements.txt
Run the model training script:
python main.py

```

📈 Accuracy Achieved
Training Accuracy: ~99%

Test Accuracy: ~97%

Your results may vary depending on architecture and training time.

💡 Future Improvements
Add custom digit image upload & prediction

Build a web app with Streamlit or Flask

Try CNNs for even better accuracy

👩🏽‍💻 Author
Valerie Onoja Amarachi
AI Agent & ML Engineer in the Making 🌱
GitHub: github.com/LerieV | LinkedIn: linkedin.com/in/valerie-onoja-9828592b6/

📝 License
MIT License. Feel free to use, improve, and share!
