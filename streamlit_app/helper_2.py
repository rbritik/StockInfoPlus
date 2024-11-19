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

def predict_sentiment(articles_data, model,tokenizer):
    sentiment_scores = []
    
    for article in articles_data:
        text_to_analyze = article['title'] + " " + article['content']
        print(text_to_analyze)
        inputs = tokenizer(text_to_analyze, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()
        print(predicted_label)
        
        sentiment_scores.append(predicted_label)
        
    if sentiment_scores:
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        return average_sentiment
    else:
        return None
    





