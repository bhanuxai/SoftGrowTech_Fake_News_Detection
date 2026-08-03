from utils.prediction import predict_news

news = input("Enter News:\n\n")

result = predict_news(news)

print("\nPrediction :", result["prediction"])
print("Confidence :", result["confidence"], "%")