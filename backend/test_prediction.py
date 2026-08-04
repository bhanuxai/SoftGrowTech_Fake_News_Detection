from utils.prediction import predict_news

news = """
WASHINGTON (Reuters) - The U.S. Senate voted on Tuesday to approve a new infrastructure funding bill after weeks of bipartisan negotiations. Officials said the legislation would improve transportation and public services across the country.
"""

result = predict_news(news)

print(result)