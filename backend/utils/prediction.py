import os
import joblib

from utils.preprocessing import clean_text

# -----------------------------
# Load Saved Model & Vectorizer
# -----------------------------

from config import Config

MODEL_PATH = Config.MODEL_PATH
VECTORIZER_PATH = Config.VECTORIZER_PATH

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

    probabilities = model.predict_proba(vectorized_text)[0]

    fake_probability = float(round(probabilities[0] * 100, 2))
    real_probability = float(round(probabilities[1] * 100, 2))

    return {
        "prediction": "Real" if prediction == 1 else "Fake",
        "confidence": max(fake_probability, real_probability),
        "probabilities": {
            "fake": fake_probability,
            "real": real_probability
        }
    }