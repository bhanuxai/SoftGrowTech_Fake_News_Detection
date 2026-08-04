# 📰 SoftGrowTech Fake News Detection System

> An AI-powered Fake News Detection System developed during the **SoftGrowTech Machine Learning Internship** using **Natural Language Processing (NLP)** and **Machine Learning**.

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST%20API-black?style=for-the-badge&logo=flask)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-CSS-38B2AC?style=for-the-badge&logo=tailwindcss)

---

# 📌 Project Overview

The **SoftGrowTech Fake News Detection System** is an intelligent web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP) techniques and Machine Learning.

The application preprocesses textual news content, converts it into numerical representations using **TF-IDF Vectorization**, and predicts authenticity using a trained **Logistic Regression** classifier.

The project exposes the trained model through a **Flask REST API** and is designed to integrate seamlessly with a modern React frontend.

---

# 🚀 Features

- 📰 Detect Fake and Real News
- 🧠 Natural Language Processing (NLP)
- 🔤 TF-IDF Feature Extraction
- 🤖 Logistic Regression Classifier
- 🌐 Flask REST API
- ⚛️ React Frontend
- 📊 Confidence Score
- 📈 Prediction Probability
- 📱 Responsive User Interface
- ⚡ Fast Predictions

---

# 🛠 Tech Stack

## Backend

- Python
- Flask
- Flask-CORS

## Machine Learning

- Scikit-learn
- Logistic Regression
- TF-IDF Vectorizer

## NLP

- NLTK
- Stopwords Removal
- Lemmatization
- Text Cleaning

## Frontend

- React.js
- Tailwind CSS
- Axios
- Vite

---

# 📂 Project Structure

```text
SoftGrowTech_Fake_News_Detection/

│
├── backend/
│   │
│   ├── dataset/
│   │   ├── Fake.csv
│   │   └── True.csv
│   │
│   ├── models/
│   │   ├── fake_news_model.pkl
│   │   └── tfidf_vectorizer.pkl
│   │
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── prediction.py
│   │   └── __init__.py
│   │
│   ├── app.py
│   ├── train_model.py
│   ├── test_prediction.py
│   ├── requirements.txt
│   └── config.py
│
├── frontend/
│
└── README.md
```

---

# 🧠 Machine Learning Workflow

```
News Article
      │
      ▼
Text Preprocessing
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Lemmatization
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Logistic Regression
      │
      ▼
Prediction
```

---

# 📊 Model Information

| Component | Algorithm |
|-----------|-----------|
| Feature Extraction | TF-IDF Vectorizer |
| Classification | Logistic Regression |
| NLP Library | NLTK |
| Framework | Scikit-Learn |

---

# 🌐 API Endpoints

## Health Check

```
GET /
```

Response

```json
{
    "success": true,
    "message": "🚀 Fake News Detection API is running successfully!"
}
```

---

## Predict News

```
POST /predict
```

### Request

```json
{
    "text": "NASA successfully launched a new satellite into Earth's orbit."
}
```

### Response

```json
{
    "success": true,
    "prediction": "Real",
    "confidence": 99.84,
    "probabilities": {
        "real": 99.84,
        "fake": 0.16
    }
}
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/SoftGrowTech_Fake_News_Detection.git
```

---

## Backend Setup

```bash
cd backend
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run Backend

```bash
python app.py
```

Backend starts at

```
http://127.0.0.1:5000
```

---

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

# 📸 Screenshots

## Home Page

> *(Add Screenshot Here)*

---

## Prediction Result

> *(Add Screenshot Here)*

---

## API Testing

> *(Add Thunder Client Screenshot Here)*

---

# 📈 Future Improvements

- Transformer-based Models (BERT)
- Explainable AI (SHAP/LIME)
- Multi-language Fake News Detection
- News Source Credibility Analysis
- Cloud Deployment
- User Authentication
- Prediction History
- Admin Dashboard

---

# 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Natural Language Processing (NLP)
- Text Cleaning
- TF-IDF Vectorization
- Machine Learning Classification
- Flask REST API Development
- React Integration
- Model Deployment
- API Development
- Git & GitHub Workflow

---

# 👨‍💻 Developed By

**Bhanu Sesha Sai**

B.Tech CSE (AI & ML)

Machine Learning Intern @ **SoftGrowTech**

---

# 📜 License

This project was developed for educational and internship purposes.

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub!
