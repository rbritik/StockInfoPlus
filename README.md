# Stock News Analysis Application 📰

## Overview
The **Stock News Analysis Application** provides insights into the latest news about selected stocks and performs sentiment analysis on the news articles. Built with **Streamlit**, this interactive app fetches real-time news, evaluates their sentiment using a pre-trained transformer model, and displays whether the news is positive or negative.

---

## Features
- Fetches real-time news articles for a user-selected stock.
- Displays article details including:
  - **Source**
  - **Title**
  - **Description**
  - **Content**
- Performs sentiment analysis using the `distilbert-base-uncased-finetuned-sst-2-english` model.
- Provides actionable feedback:
  - **Positive sentiment**: Encouraging message.
  - **Negative sentiment**: Cautionary message.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-news-analysis.git
   cd stock-news-analysis
