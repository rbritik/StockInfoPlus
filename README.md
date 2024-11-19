# Stock Info Plus 📰📈

## Overview
**Stock Info Plus** is an advanced web-based application designed to analyze stocks through **technical**, **fundamental**, and **sentiment analysis**. The app combines real-time data visualization, powerful sentiment evaluation, and key stock insights to empower investors with actionable intelligence.

---

## Features

### 1. **Technical Analysis**
- Visualizes key technical indicators:
  - **Relative Strength Index (RSI):** Helps identify overbought or oversold conditions.
  - **Simple Moving Average (SMA):** Tracks stock trends over different periods (e.g., 50-day and 100-day SMAs).
  - **Exponential Moving Average (EMA):** Tracks price movements with recent data weighted more heavily.
- Highlights buy/sell signals using a **crossover system** of 50-day and 100-day SMAs.

### 2. **Fundamental Analysis**
- Provides real-time news articles for selected stocks using **NewsAPI**.
- Allows users to explore:
  - **Source**
  - **Title**
  - **Description**
  - **Content** of each news article.

### 3. **Sentiment Analysis**
- Leverages a pre-trained transformer model (`distilbert-base-uncased-finetuned-sst-2-english`) for evaluating news sentiment.
- Aggregates sentiment scores to classify news as:
  - **Positive News**: Displays encouraging messages and insights.
  - **Negative News**: Cautionary messages are provided for informed decision-making.

---

## Installation

### Prerequisites
- Python 3.7 or above.
- API key for **NewsAPI** (obtainable from [NewsAPI](https://newsapi.org)).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-info-plus.git
   cd stock-info-plus
