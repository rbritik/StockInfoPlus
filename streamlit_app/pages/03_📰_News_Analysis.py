import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newsapi import NewsApiClient
import streamlit as st
from helper_2 import *
from helper import *

# Configure the page
st.set_page_config(
    page_title="Stock News Analysis",
    page_icon="📰",
)

st.markdown("# **Stock News Analysis**")

# Sidebar for user inputs
st.sidebar.markdown("## **User Input Features**")
stock_dict = fetch_stocks()
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key='62c1aa630ad84904950514ff553a39f5')
company_name = f"{stock}"
num_articles = 5
news_data = fetch_news_data(company_name, num_articles)

# Display the news articles
for idx, article in enumerate(news_data, start=1):
    st.subheader(f"Article {idx}")
    st.write("**Source:**", article['source'])
    st.write("**Title:**", article['title'])
    st.write("**Description:**", article['description'])
    st.write("**Content:**", article['content'])
    st.markdown("---")  # Horizontal line

# Sentiment Analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

average_sentiment = predict_sentiment(news_data, model, tokenizer)

# Display the sentiment score and message
if average_sentiment is not None:
    st.markdown(f"### **Sentiment Score:** {average_sentiment:.2f}")
    if average_sentiment > 0.7:
        st.success("🌟 The news sentiment is highly positive! This might be a good time to look into this stock.")
    else:
        st.error("⚠️ The news sentiment is negative. Proceed with caution before making any decisions.")
else:
    st.warning("No articles found for sentiment analysis.")
