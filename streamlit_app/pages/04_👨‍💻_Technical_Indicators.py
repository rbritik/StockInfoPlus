import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from helper import *

# Streamlit Page Configuration
st.set_page_config(
    page_title="Technical Indicators",
    page_icon="📊",
    layout="wide"
)

# Functions to calculate technical indicators
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(data, window=20):
    sma = data.rolling(window=window).mean()
    return sma

def calculate_ema(data, span=20):
    ema = data.ewm(span=span, adjust=False).mean()
    return ema

# Load data
st.title("Stock Technical Indicators Dashboard")
st.sidebar.markdown("## **User Input Features**")

# Fetch and store the stock data
stock_dict = fetch_stocks()

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", list(stock_dict.keys()))

st.sidebar.markdown("### **Select stock exchange**")
stock_exchange = st.sidebar.radio("Choose a stock exchange", ("BSE", "NSE"), index=0)

# Build the stock ticker
stock_ticker = f"{stock_dict[stock]}.{'BO' if stock_exchange == 'BSE' else 'NS'}"

st.sidebar.markdown("### **Stock ticker**")
st.sidebar.text_input(
    label="Stock ticker code", placeholder=stock_ticker, disabled=True
)

# Fetch and store periods and intervals
periods = fetch_periods_intervals()

# Add a selector for period
st.sidebar.markdown("### **Select period**")
period = st.sidebar.selectbox("Choose a period", list(periods.keys()))

# Add a selector for interval
st.sidebar.markdown("### **Select interval**")
interval = st.sidebar.selectbox("Choose an interval", periods[period])


if stock is not None:
    
    df = fetch_stock_history(stock_ticker, period, interval)
    # Calculate indicators
    Close = df['Close']
    df['RSI'] = calculate_rsi(Close)
    df['SMA'] = calculate_sma(Close)
    df['EMA'] = calculate_ema(Close)

    # --- Graph 1: Close Price and RSI ---
    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])

    # Add Close Price
    fig1.add_trace(go.Scatter(x=df.index, y=Close, name='Close Price', line=dict(color='blue')), row=1, col=1)

    # Add RSI
    fig1.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='red')), row=2, col=1)

    # Add layout settings
    fig1.update_layout(
        title="Close Price and RSI",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="RSI",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
    )

    st.plotly_chart(fig1, use_container_width=True)

    # --- Graph 2: Closing Price with 50-Day SMA ---
    sma_50 = calculate_sma(Close, window=50)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=Close, name='Close Price'))
    fig2.add_trace(go.Scatter(x=df.index, y=sma_50, name='50-Day SMA', line=dict(color='red')))

    fig2.update_layout(
        title="Stock Closing Price with 50-Day SMA",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
    )

    st.plotly_chart(fig2, use_container_width=True)

    # --- Graph 3: Moving Average Crossover System ---
    sma_100 = calculate_sma(Close, window=100)

    # Crossover System
    cross_up = (sma_50 > sma_100) & (sma_50.shift(1) < sma_100.shift(1))
    cross_down = (sma_50 < sma_100) & (sma_50.shift(1) > sma_100.shift(1))

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=Close, name='Close Price'))
    fig3.add_trace(go.Scatter(x=df.index, y=sma_50, name='50-Day SMA', line=dict(color='red')))
    fig3.add_trace(go.Scatter(x=df.index, y=sma_100, name='100-Day SMA', line=dict(color='green')))

    # Add Buy and Sell signals
    fig3.add_trace(go.Scatter(
        x=df.index[cross_up],
        y=df['Close'][cross_up],
        mode='markers',
        marker=dict(color='green', symbol='triangle-up', size=10),
        name='Buy Signal'
    ))

    fig3.add_trace(go.Scatter(
        x=df.index[cross_down],
        y=df['Close'][cross_down],
        mode='markers',
        marker=dict(color='red', symbol='triangle-down', size=10),
        name='Sell Signal'
    ))

    fig3.update_layout(
        title="Stock Closing Price with Moving Average Crossover System",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1)
    )

    st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Please upload a CSV file to analyze.")
