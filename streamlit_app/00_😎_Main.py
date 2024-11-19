import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Stock Info Plus",
    page_icon="📈",
    layout="centered"
)

# Main landing page content
st.title("📊 Welcome to Stock Info Plus!")
st.markdown(
    """
    **Stock Info Plus** is your one-stop solution for comprehensive stock market analysis and predictions. 
    Our app provides you with real-time insights, technical indicators, and intelligent predictions 
    to help you make informed decisions in the stock market.
    """
)

# Key features
st.subheader("✨ Key Features")
st.markdown(
    """
    - 📉 **Technical Analysis:** Visualize RSI, SMA, EMA, and moving average crossovers.
    - 📈 **Prediction Models:** Leverage advanced analytics to forecast stock trends.
    - 🗞️ **Sentiment Analysis:** Analyze news articles to understand market sentiment.
    - 🚀 **Interactive Visualizations:** Dive deep into interactive and dynamic stock data visualizations.
    """
)

# Call to action
st.divider()
st.markdown(
    """
    #### Ready to explore the world of stock insights? 
    Use the navigation menu on the left to get started with:
    - 📊 **Stock Analysis**
    - 📰 **News Sentiment**
    - 🔮 **Prediction Models**
    """
)
st.divider()

# Add an inspiring footer or tagline
st.markdown(
    """
    ---
    💡 *"Invest smartly with data-driven insights from Stock Info Plus!"*
    """
)
