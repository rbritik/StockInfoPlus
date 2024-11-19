
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsapi import NewsApiClient

# Initialize NewsApiClient with your API key
newsapi = NewsApiClient(api_key='62c1aa630ad84904950514ff553a39f5')

def fetch_news_data(company_name, num_articles=5):
    # Define the query to search for news related to the company
    query = company_name
   

    # Fetch news articles using the News API
    news_articles = newsapi.get_everything(q=query, language='en', sort_by='relevancy', page_size=num_articles)

    # Extract relevant information from the news articles
    articles_data = []
    for article in news_articles['articles']:
        articles_data.append({
            'source': article['source']['name'],
            'author': article['author'],
            'title': article['title'],
            'description': article['description'],
            'content': article['content']
        })

    return articles_data

# company_name = "Adani"
# num_articles = 5
# news_data = fetch_news_data(company_name, num_articles)

# # Print the fetched news data
# for idx, article in enumerate(news_data, start=1):
#     print(f"Article {idx}:")
#     print("Source:", article['source'])
#     print("Title:", article['title'])
#     print("Description:", article['description'])
#     print("Content:", article['content'])
#     print()





# model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)


def predict_sentiment(articles_data, model,tokenizer):
    sentiment_scores = []
    
    for article in articles_data:
        text_to_analyze = article['title'] + " " + article['content']
        print(text_to_analyze)
        inputs = tokenizer(text_to_analyze, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
#         print(outputs.logits)
        predicted_label = torch.argmax(outputs.logits).item()
        print(predicted_label)
        
        sentiment_scores.append(predicted_label)
        
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return average_sentiment
    else:
        return None
    
# average_sentiment = predict_sentiment(news_data)

# if average_sentiment is not None:
#     print("Average Sentiment: ", average_sentiment)
# else:
#     print("No articles found.")






