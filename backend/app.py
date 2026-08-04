import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS

from utils.prediction import predict_news

app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "🚀 Fake News Detection API is running successfully!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data received."
            }), 400

        news_text = data.get("text", "").strip()

        if news_text == "":
            return jsonify({
                "success": False,
                "message": "News text cannot be empty."
            }), 400

        result = predict_news(news_text)

        return jsonify({
    "success": True,
    "prediction": result["prediction"],
    "confidence": result["confidence"],
    "probabilities": result["probabilities"]
})

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": True
    })

if __name__ == "__main__":
    app.run(debug=True)