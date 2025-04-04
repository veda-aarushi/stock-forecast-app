# ----------------------------------------
# 📊 Stock Price Forecasting Web App
# Powered by Streamlit, yFinance, Prophet, and Plotly
# ----------------------------------------

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# ----------------------------------------
# 🔧 Constants
# ----------------------------------------
START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# ----------------------------------------
# 🎯 App Configuration
# ----------------------------------------
st.set_page_config(page_title="📈 Stock Forecast App", layout="centered")
st.title("📈 Stock Forecast App")
st.markdown("""
This app allows you to **forecast stock prices** using Facebook Prophet.  
Select a stock, choose how far ahead you want to forecast, and visualize trends and seasonality in the market.
""")

# ----------------------------------------
# 🧠 User Input
# ----------------------------------------
available_stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox("🔍 Choose a stock to forecast:", available_stocks)

n_years = st.slider("📅 Forecast window (years):", 1, 4)
forecast_period = n_years * 365  # Days to predict

# ----------------------------------------
# 📥 Load Stock Data
# ----------------------------------------
@st.cache_data
def load_stock_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data

with st.spinner("Loading historical stock data..."):
    data = load_stock_data(selected_stock)
st.success("Data loaded successfully!")

# ----------------------------------------
# 🧾 Display Raw Data
# ----------------------------------------
st.subheader("🔢 Latest Historical Data")
st.dataframe(data.tail())

# ----------------------------------------
# 📈 Plot Raw Time Series
# ----------------------------------------
def plot_time_series():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
    fig.update_layout(
        title="📊 Stock Prices Over Time",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig)

plot_time_series()

# ----------------------------------------
# 🔮 Forecasting with Prophet
# ----------------------------------------
# Step 1: Prepare data for Prophet
df_prophet = data[["Date", "Close"]].copy()
df_prophet.columns = ["ds", "y"]
df_prophet["y"] = pd.to_numeric(df_prophet["y"], errors="coerce")
df_prophet.dropna(inplace=True)

# Step 2: Train the model
model = Prophet()
model.fit(df_prophet)

# Step 3: Make future predictions
future_df = model.make_future_dataframe(periods=forecast_period)
forecast = model.predict(future_df)

# ----------------------------------------
# 📉 Show Forecast Table
# ----------------------------------------
st.subheader("🧮 Forecast Data")
st.dataframe(forecast.tail())

# ----------------------------------------
# 📅 Forecast Plot
# ----------------------------------------
st.subheader(f"📈 Forecast Plot for {n_years} Year(s)")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast)

# ----------------------------------------
# 🔬 Forecast Components
# ----------------------------------------
st.subheader("🔍 Forecast Decomposition")
fig_components = model.plot_components(forecast)
st.write(fig_components)
# ----------------------------------------
# 🤖 Price Movement Classifier (Up/Down)
# ----------------------------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.sidebar.subheader("🔍 Predict: Will price go up tomorrow?")
if st.sidebar.button("Run Classifier"):
    # -------------------------------
    # 🏗️ Feature Engineering
    # -------------------------------
    df_class = data[["Date", "Close"]].copy()
    df_class["Prev Close"] = df_class["Close"].shift(1)
    df_class["Daily Change %"] = df_class["Close"].pct_change() * 100
    df_class["Target"] = (df_class["Close"].shift(-1) > df_class["Close"]).astype(int)

    # Clean and drop NaNs
    df_class.dropna(inplace=True)

    # -------------------------------
    # 📊 Model Training
    # -------------------------------
    features = df_class[["Prev Close", "Daily Change %"]]
    target = df_class["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # -------------------------------
    # 📈 Evaluate Accuracy
    # -------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Classifier Performance")
    st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

    # -------------------------------
    # 🔮 Predict for "Tomorrow"
    # -------------------------------
    latest_data = features.iloc[-1].values.reshape(1, -1)
    tomorrow_prediction = model.predict(latest_data)[0]
    prediction_label = "📈 Yes, likely to go UP" if tomorrow_prediction == 1 else "📉 No, likely to go DOWN"

    st.subheader("🤖 Prediction for Tomorrow")
    st.write(f"Will the stock price of **{selected_stock}** go up tomorrow?")
    st.success(prediction_label)
