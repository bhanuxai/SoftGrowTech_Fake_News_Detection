import os
import joblib

from utils.preprocessing import clean_text

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------

MODEL_PATH = os.path.join("models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_news(news_text):
    """
    Predict whether the given news is Fake or Real.

    Returns:
        dict
    """

    cleaned_text = clean_text(news_text)

    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    confidence = model.predict_proba(vectorized_text).max()

    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": round(confidence * 100, 2)
    }