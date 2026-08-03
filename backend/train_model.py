import os
import pandas as pd

# Import the preprocessing function
from utils.preprocessing import clean_text

# -----------------------------
# Load Dataset
# -----------------------------

fake_path = os.path.join("dataset", "Fake.csv")
true_path = os.path.join("dataset", "True.csv")

fake_df = pd.read_csv(fake_path)
true_df = pd.read_csv(true_path)

print("=" * 50)
print("Datasets Loaded Successfully")
print("=" * 50)

print(f"Fake News Articles : {len(fake_df)}")
print(f"Real News Articles : {len(true_df)}")

print("\nFake Dataset Preview:")
print(fake_df.head())

print("\nReal Dataset Preview:")
print(true_df.head())


# -----------------------------
# Add Labels
# -----------------------------

fake_df["label"] = 0   # Fake News
true_df["label"] = 1   # Real News

print("\nLabels Added Successfully!")

# -----------------------------
# Merge Both Datasets
# -----------------------------

df = pd.concat([fake_df, true_df], ignore_index=True)

print(f"\nTotal Articles: {len(df)}")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Shuffled Successfully!")

# -----------------------------
# Combine Title and Text
# -----------------------------

df["content"] = df["title"] + " " + df["text"]

print("\nCleaning text... Please wait.")

# Apply preprocessing
df["content"] = df["content"].apply(clean_text)

print("Text Cleaning Completed!")

# Display sample data
print("\nSample Cleaned Data:")
print(df[["content", "label"]].head())


# -----------------------------
# Feature Extraction (TF-IDF)
# -----------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("\nConverting text into TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X = vectorizer.fit_transform(df["content"])
y = df["label"]

print("TF-IDF Vectorization Completed!")

print(f"Feature Matrix Shape: {X.shape}")

# -----------------------------
# Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDataset Split Successfully!")

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")



# -----------------------------
# Train Logistic Regression Model
# -----------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

print("\nTraining Logistic Regression Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# -----------------------------
# Model Evaluation
# -----------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model and Vectorizer
# -----------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Location : models/fake_news_model.pkl")

print("Vectorizer Saved Successfully!")
print("Location : models/tfidf_vectorizer.pkl")