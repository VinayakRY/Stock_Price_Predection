import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# --- Streamlit setup ---
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction Dashboard")

# --- User Input ---
ticker_symbol = st.text_input("Enter NSE Ticker (e.g. SBIN.NS)", "SBIN.NS")

# --- Live Stock Price Info ---
try:
    ticker_obj = yf.Ticker(ticker_symbol)
    info = ticker_obj.info

    current_price = info.get("currentPrice")
    previous_close = info.get("previousClose")
    open_price = info.get("open")
    day_high = info.get("dayHigh")
    day_low = info.get("dayLow")

    if current_price and previous_close:
        change = current_price - previous_close
        percent_change = (change / previous_close) * 100
    else:
        change = percent_change = None

    st.subheader("💹 Live Stock Price Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Current Price", f"₹{current_price:.2f}" if current_price else "N/A",
                f"{change:+.2f} ({percent_change:+.2f}%)" if change else "N/A")
    col2.metric("Previous Close", f"₹{previous_close:.2f}" if previous_close else "N/A")
    col3.metric("Open", f"₹{open_price:.2f}" if open_price else "N/A")
    col4.metric("Day High", f"₹{day_high:.2f}" if day_high else "N/A")
    col5.metric("Day Low", f"₹{day_low:.2f}" if day_low else "N/A")
    col6.metric("Symbol", ticker_symbol.upper())

except Exception as e:
    st.warning(f"⚠️ Unable to fetch live stock info: {e}")

# --- Load Historical Data ---
data_load_state = st.text("Fetching stock data...")
try:
    data = yf.download(ticker_symbol, start="2015-01-01")
    data_load_state.text("")
except Exception as e:
    st.error(f"❌ Failed to fetch stock data: {e}")
    st.stop()

if data.empty:
    st.error("❌ No stock data found. Please check the ticker symbol.")
    st.stop()

data.dropna(inplace=True)
data.reset_index(inplace=True)

# --- Feature Engineering ---
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# --- Model Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.success(f"📐 Model RMSE: {rmse:.2f}")

# --- Actual vs Predicted Chart ---
st.subheader("📊 Actual vs Predicted Candlestick Chart")
predicted_df = data.iloc[y_test.index].copy()
predicted_df['Predicted_Close'] = y_pred

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=predicted_df['Date'],
    open=predicted_df['Open'],
    high=predicted_df['High'],
    low=predicted_df['Low'],
    close=predicted_df['Close'],
    name='Actual'
))
fig.add_trace(go.Scatter(
    x=predicted_df['Date'],
    y=predicted_df['Predicted_Close'],
    mode='lines',
    name='Predicted',
    line=dict(color='orange', width=2)
))
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (₹)',
    template='plotly_dark',
    height=600,
    legend=dict(orientation='h', y=1.1)
)
st.plotly_chart(fig, use_container_width=True, key="actual_vs_predicted")

# --- Helper: Future Prediction Generator ---
def generate_future_predictions(start_row, periods, freq="D"):
    future_data = []
    last_known_data = np.array([[float(start_row['Open']), float(start_row['High']), float(start_row['Low']), float(start_row['Volume'])]])
    start_date = pd.to_datetime(start_row['Date'].item()) + timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=periods, freq=freq)

    for date in future_dates:
        last_known_data = last_known_data.reshape(1, -1)
        pred_close = float(model.predict(last_known_data).item())
        pred_open = pred_high = pred_low = pred_close
        volume = float(last_known_data[0][3])

        future_data.append({
            'Date': date,
            'Open': pred_open,
            'High': pred_high,
            'Low': pred_low,
            'Close': pred_close,
            'Volume': volume
        })

        last_known_data = np.array([[pred_open, pred_high, pred_low, volume]])

    return pd.DataFrame(future_data)

# --- Prediction Tabs ---
st.subheader("🔮 Predict Future Prices")
tab1, tab2 = st.tabs(["Next 7 Days (Auto)", "Next 7 Days from Custom Date"])

# --- Tab 1: Next 7 Days (Automatic) ---
with tab1:
    st.markdown("📅 **Auto Prediction: Next 7 Days from Today**")
    last_row = data.iloc[[-1]].copy()
    last_row['Date'] = pd.to_datetime(last_row['Date'])

    future_df = generate_future_predictions(last_row.iloc[0], 7)
    st.dataframe(future_df[['Date', 'Close']].rename(columns={'Close': 'Predicted Close'}).style.format({'Predicted Close': '₹{:.2f}'}))

    fig_future = go.Figure(data=[go.Candlestick(
        x=future_df['Date'],
        open=future_df['Open'],
        high=future_df['High'],
        low=future_df['Low'],
        close=future_df['Close'],
        name="Predicted Future"
    )])
    fig_future.update_layout(
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        template='plotly_dark',
        height=500
    )
    st.plotly_chart(fig_future, use_container_width=True, key="future_auto")

# --- Tab 2: Manual Date Input (Future Only) ---
with tab2:
    st.markdown("📅 **Manual Prediction: Choose a Future Date**")
    user_date = st.date_input("Select a future date (≥ today):", datetime.today().date())

    if user_date < datetime.today().date():
        st.warning("⚠️ Please select today or a future date only.")
    else:
        last_known_row = data.iloc[[-1]].copy()
        last_known_row['Date'] = pd.to_datetime(user_date)

        predicted_df = generate_future_predictions(last_known_row.iloc[0], 7)
        st.dataframe(predicted_df[['Date', 'Close']].rename(columns={'Close': 'Predicted Close'}).style.format({'Predicted Close': '₹{:.2f}'}))

        fig_custom = go.Figure(data=[go.Candlestick(
            x=predicted_df['Date'],
            open=predicted_df['Open'],
            high=predicted_df['High'],
            low=predicted_df['Low'],
            close=predicted_df['Close'],
            name="Predicted Future"
        )])
        fig_custom.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig_custom, use_container_width=True, key="future_manual")

# --- Disclaimer ---
st.info(
    "⚠️ These predictions are generated using a basic linear regression model trained on historical Open, High, Low, and Volume values. "
    "They are for educational/demo purposes and not suitable for real trading decisions."
)

