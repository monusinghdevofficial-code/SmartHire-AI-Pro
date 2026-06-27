# 🚀 SmartHire AI Pro

Deep Learning Powered Resume Screening System built using CNN, TensorFlow, Streamlit and NLP.

## 📌 Overview

SmartHire AI Pro is an intelligent resume screening system that automatically analyzes resumes and predicts the most relevant job category using a trained CNN (Convolutional Neural Network) model.

The system extracts text from PDF resumes, identifies technical skills, calculates a resume score, provides ATS analysis, and predicts the candidate category with confidence score.

---

## ✨ Features

* 📄 PDF Resume Upload
* 🤖 CNN-Based Resume Classification
* 🎯 Category Prediction
* 📊 Confidence Score
* 🔍 Skill Extraction
* 📈 Resume Scoring System
* ✅ ATS Resume Analysis
* ⚠️ Low Confidence Warning
* 🎨 Professional Streamlit UI

---

## 🧠 Machine Learning Pipeline

### Dataset

* Resume Dataset from Kaggle
* 2484 Resume Samples
* 24 Different Categories

### Data Preprocessing

* Text Cleaning
* Tokenization
* Label Encoding
* Sequence Padding

### Deep Learning Model

* Embedding Layer
* Conv1D Layer
* Global Max Pooling
* Dense Layers
* Dropout Layers
* Softmax Output Layer

### Model Performance

| Model             | Accuracy |
| ----------------- | -------- |
| Decision Tree     | ~63%     |
| Random Forest     | ~69%     |
| Naive Bayes       | ~56%     |
| XGBoost           | ~77%     |
| CNN (Final Model) | ~80.68%  |

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### Deep Learning

* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas

### NLP

* Tokenizer
* Label Encoder

### PDF Processing

* PyPDF2

---

## 📂 Project Structure

SmartHire_AI/

├── app.py

├── resume_cnn_model.h5

├── tokenizer.pkl

├── label_encoder.pkl

├── requirements.txt

├── README.md

└── sample_resume.pdf

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/SmartHire-AI-Pro.git
cd SmartHire-AI-Pro
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📊 Output

The application provides:

* Resume Score
* Predicted Category
* Confidence Percentage
* Detected Skills
* ATS Analysis
* AI Recommendation

---

## 🎯 Future Enhancements

* Resume vs Job Description Matching
* ATS Score Breakdown
* Skill Gap Analysis
* PDF Report Generation
* AI Career Recommendation
* LLM-Based Resume Feedback

---

## 👨‍💻 Author

Monu Singh

B.Tech CSE (AI & ML)

SmartHire AI Pro Developer

---

## ⭐ Support

If you like this project, please give it a star on GitHub.
