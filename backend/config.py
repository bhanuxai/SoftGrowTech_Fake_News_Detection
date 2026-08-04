import os


class Config:
    DEBUG = True

    MODEL_PATH = os.path.join("models", "fake_news_model.pkl")

    VECTORIZER_PATH = os.path.join("models", "tfidf_vectorizer.pkl")