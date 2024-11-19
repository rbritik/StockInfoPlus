# Imports
import datetime as dt
import os
from pathlib import Path

import ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Import pandas
import pandas as pd

# Import yfinance
import yfinance as yf

# Import the required libraries
from statsmodels.tsa.ar_model import AutoReg


# Create function to fetch stock name and id
def fetch_stocks():
    # Load the data
    df = pd.read_csv(Path.cwd() / "data" / "equity_issuers.csv")

    # Filter the data
    df = df[["Security Code", "Issuer Name"]]

    # Create a dictionary
    stock_dict = dict(zip(df["Security Code"], df["Issuer Name"]))

    # Return the dictionary
    return stock_dict


# Create function to fetch periods and intervals
def fetch_periods_intervals():
    # Create dictionary for periods and intervals
    periods = {
        "1d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "5d": ["1m", "2m", "5m", "15m", "30m", "60m", "90m"],
        "1mo": ["30m", "60m", "90m", "1d"],
        "3mo": ["1d", "5d", "1wk", "1mo"],
        "6mo": ["1d", "5d", "1wk", "1mo"],
        "1y": ["1d", "5d", "1wk", "1mo"],
        "2y": ["1d", "5d", "1wk", "1mo"],
        "5y": ["1d", "5d", "1wk", "1mo"],
        "10y": ["1d", "5d", "1wk", "1mo"],
        "max": ["1d", "5d", "1wk", "1mo"],
    }

    # Return the dictionary
    return periods


# Function to fetch the stock info
def fetch_stock_info(stock_ticker):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_info = stock_data.info

    # Function to safely get value from dictionary or return "N/A"
    def safe_get(data_dict, key):
        return data_dict.get(key, "N/A")

    # Extract only the important information
    stock_data_info = {
        "Basic Information": {
            "symbol": safe_get(stock_data_info, "symbol"),
            "longName": safe_get(stock_data_info, "longName"),
            "currency": safe_get(stock_data_info, "currency"),
            "exchange": safe_get(stock_data_info, "exchange"),
        },
        "Market Data": {
            "currentPrice": safe_get(stock_data_info, "currentPrice"),
            "previousClose": safe_get(stock_data_info, "previousClose"),
            "open": safe_get(stock_data_info, "open"),
            "dayLow": safe_get(stock_data_info, "dayLow"),
            "dayHigh": safe_get(stock_data_info, "dayHigh"),
            "regularMarketPreviousClose": safe_get(
                stock_data_info, "regularMarketPreviousClose"
            ),
            "regularMarketOpen": safe_get(stock_data_info, "regularMarketOpen"),
            "regularMarketDayLow": safe_get(stock_data_info, "regularMarketDayLow"),
            "regularMarketDayHigh": safe_get(stock_data_info, "regularMarketDayHigh"),
            "fiftyTwoWeekLow": safe_get(stock_data_info, "fiftyTwoWeekLow"),
            "fiftyTwoWeekHigh": safe_get(stock_data_info, "fiftyTwoWeekHigh"),
            "fiftyDayAverage": safe_get(stock_data_info, "fiftyDayAverage"),
            "twoHundredDayAverage": safe_get(stock_data_info, "twoHundredDayAverage"),
        },
        "Volume and Shares": {
            "volume": safe_get(stock_data_info, "volume"),
            "regularMarketVolume": safe_get(stock_data_info, "regularMarketVolume"),
            "averageVolume": safe_get(stock_data_info, "averageVolume"),
            "averageVolume10days": safe_get(stock_data_info, "averageVolume10days"),
            "averageDailyVolume10Day": safe_get(
                stock_data_info, "averageDailyVolume10Day"
            ),
            "sharesOutstanding": safe_get(stock_data_info, "sharesOutstanding"),
            "impliedSharesOutstanding": safe_get(
                stock_data_info, "impliedSharesOutstanding"
            ),
            "floatShares": safe_get(stock_data_info, "floatShares"),
        },
        "Dividends and Yield": {
            "dividendRate": safe_get(stock_data_info, "dividendRate"),
            "dividendYield": safe_get(stock_data_info, "dividendYield"),
            "payoutRatio": safe_get(stock_data_info, "payoutRatio"),
        },
        "Valuation and Ratios": {
            "marketCap": safe_get(stock_data_info, "marketCap"),
            "enterpriseValue": safe_get(stock_data_info, "enterpriseValue"),
            "priceToBook": safe_get(stock_data_info, "priceToBook"),
            "debtToEquity": safe_get(stock_data_info, "debtToEquity"),
            "grossMargins": safe_get(stock_data_info, "grossMargins"),
            "profitMargins": safe_get(stock_data_info, "profitMargins"),
        },
        "Financial Performance": {
            "totalRevenue": safe_get(stock_data_info, "totalRevenue"),
            "revenuePerShare": safe_get(stock_data_info, "revenuePerShare"),
            "totalCash": safe_get(stock_data_info, "totalCash"),
            "totalCashPerShare": safe_get(stock_data_info, "totalCashPerShare"),
            "totalDebt": safe_get(stock_data_info, "totalDebt"),
            "earningsGrowth": safe_get(stock_data_info, "earningsGrowth"),
            "revenueGrowth": safe_get(stock_data_info, "revenueGrowth"),
            "returnOnAssets": safe_get(stock_data_info, "returnOnAssets"),
            "returnOnEquity": safe_get(stock_data_info, "returnOnEquity"),
        },
        "Cash Flow": {
            "freeCashflow": safe_get(stock_data_info, "freeCashflow"),
            "operatingCashflow": safe_get(stock_data_info, "operatingCashflow"),
        },
        "Analyst Targets": {
            "targetHighPrice": safe_get(stock_data_info, "targetHighPrice"),
            "targetLowPrice": safe_get(stock_data_info, "targetLowPrice"),
            "targetMeanPrice": safe_get(stock_data_info, "targetMeanPrice"),
            "targetMedianPrice": safe_get(stock_data_info, "targetMedianPrice"),
        },
    }

    # Return the stock data
    return stock_data_info


# Function to fetch the stock history
def fetch_stock_history(stock_ticker, period, interval):
    # Pull the data for the first security
    stock_data = yf.Ticker(stock_ticker)

    # Extract full of the stock
    stock_data_history = stock_data.history(period=period, interval=interval)[
        ["Open", "High", "Low", "Close"]
    ]

    # Return the stock data
    return stock_data_history


# Function to generate the stock prediction
def generate_stock_prediction(stock_ticker):
    # Try to generate the predictions
    try:
        # Pull the data for the first security
        stock_data = yf.Ticker(stock_ticker)

        # Extract the data for last 1yr with 1d interval
        stock_data_hist = stock_data.history(period="2y", interval="1d")
        # print(stock_data_hist)

        # Clean the data for to keep only the required columns
        stock_data_close = stock_data_hist[["Close"]]

        # Change frequency to day
        stock_data_close = stock_data_close.asfreq("D", method="ffill")

        # Fill missing values
        stock_data_close = stock_data_close.ffill()

        # Define training and testing area
        train_df = stock_data_close.iloc[: int(len(stock_data_close) * 0.9) + 1]  # 90%
        test_df = stock_data_close.iloc[int(len(stock_data_close) * 0.9) :]  # 10%

        # Define training model
        model = AutoReg(train_df["Close"], 250).fit(cov_type="HC0")

        # Predict data for test data
        predictions = model.predict(
            start=test_df.index[0], end=test_df.index[-1], dynamic=True
        )

        # Predict 90 days into the future
        forecast = model.predict(
            start=test_df.index[0],
            end=test_df.index[-1] + dt.timedelta(days=90),
            dynamic=True,
        )

        # Return the required data
        return train_df, test_df, forecast, predictions

    # If error occurs
    except:
        # Return None
        return None, None, None, None

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


def identify_single_candlestick_pattern(open_price, close_price, high_price, low_price):
    if open_price > close_price:
        body_color = 'red'
    else:
        body_color = 'green'

    body_height = abs(close_price - open_price)
    upper_shadow_height = high_price - max(open_price, close_price)
    lower_shadow_height = min(open_price, close_price) - low_price

    # Long body indicates strong buying or selling pressure
    if body_height > (high_price - low_price) * 0.5:
        if upper_shadow_height < body_height * 0.1 and lower_shadow_height < body_height * 0.1:
            if body_color == 'green':
                return 1  
            else:
                return -1  

    # Doji
    if body_height < (high_price - low_price) * 0.1:
        return 0  

    # Hammer
    if body_color == 'green' and upper_shadow_height > body_height * 2 and lower_shadow_height < body_height * 0.1:
        return 1 

    # Hanging Man
    if body_color == 'red' and upper_shadow_height > body_height * 2 and lower_shadow_height < body_height * 0.1:
        return -1  

    # Inverted Hammer
    if body_color == 'red' and lower_shadow_height > body_height * 2 and upper_shadow_height < body_height * 0.1:
        return 1  

    # Shooting Star
    if body_color == 'red' and upper_shadow_height > body_height * 2 and lower_shadow_height < body_height * 0.1:
        return -1
    
    return 0  # No Pattern Identified

def combined_stock_prediction(stock_ticker):
    # Try to generate the predictions
    try:
        # Pull the data for the first security
        stock_data = yf.Ticker(stock_ticker)
        # print("########hello######")

        # Extract the data for last 1yr with 1d interval
        stock_data_hist = stock_data.history(period="2y", interval="1d")

        stock_data_hist['Candlestick_Pattern'] = stock_data_hist.apply(lambda row: identify_single_candlestick_pattern(row['Open'], row['Close'], row['High'], row['Low']), axis=1)
        
        stock_data_hist['ADX'] = ta.trend.ADXIndicator(stock_data_hist['High'], stock_data_hist['Low'], stock_data_hist['Close'], window=14).adx()

        stock_data_hist['RSI'] = calculate_rsi(stock_data_hist['Close'])
        stock_data_hist['SMA'] = calculate_sma(stock_data_hist['Close'])
        stock_data_hist['EMA'] = calculate_ema(stock_data_hist['Close'])
        
        selected_columns = ['Close', 'Volume', 'ADX', 'RSI', 'SMA', 'EMA', 'Candlestick_Pattern']
        
        stock_data_hist = stock_data_hist.dropna()

        stock_data_hist = stock_data_hist[selected_columns]
        
        # Define training and testing area
        training_data_len = int(np.ceil(len(stock_data_hist) * 0.70))
        
        train_df = stock_data_hist.iloc[:training_data_len]
        test_df = stock_data_hist.iloc[training_data_len:]
    
        train_data = train_df.values
        test_data = test_df.values
        
        scaler = MinMaxScaler(feature_range=(0,1))
        train_data = scaler.fit_transform(train_data)

    
        x_train = train_data[0:-1, :]
        y_train = train_data[1:,0]
        
        x_train, y_train = np.array(x_train), np.array(y_train)

        model = LinearRegression()
        x_train = x_train.reshape(x_train.shape[0], -1)
        
        model.fit(x_train, y_train)

        predictions = model.predict(x_train)

        mse = mean_squared_error(y_train, predictions)

        test_data = scaler.transform(test_data)
        x_test = test_data[:-1, :]
        y_test = test_data[1:, 0]

        x_test = np.array(x_test)

        # Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

        predictions = model.predict(x_test)

        # Combine predictions and y_test with dummy values for the other columns
        predictions_full = np.zeros_like(x_test)
        predictions_full[:, 0] = predictions

        y_test_full = np.zeros_like(x_test)
        y_test_full[:, 0] = y_test

        # Inverse transform predictions and y_test
        predictions = scaler.inverse_transform(predictions_full)[:, 0]
        y_test = scaler.inverse_transform(y_test_full)[:, 0]

        # Calculate RMSE
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
        
        
        train_data = scaler.inverse_transform(train_data)
        train_df = pd.DataFrame(train_data, columns=selected_columns, index=train_df.index)

        test_data = scaler.inverse_transform(test_data)
        test_df = pd.DataFrame(test_data, columns=selected_columns, index=test_df.index)

        # Return the required data
        return train_df, test_df, predictions
    # If error occurs
    except:
        # Return None
        return None, None, None