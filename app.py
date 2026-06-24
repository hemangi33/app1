import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="ARIMA Stock Forecast", layout="wide")

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input("Enter Stock Ticker", "AAPL")

if st.button("Forecast"):

    # Download 5 years of data
    stock = yf.download(
        ticker,
        start="2021-01-01",
        end=datetime.today().strftime('%Y-%m-%d')
    )

    if stock.empty:
        st.error("Invalid ticker symbol.")
    else:

        close_prices = stock['Close']

        st.subheader("Historical Stock Prices (Last 5 Years)")

        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(close_prices.index, close_prices.values)
        ax.set_xlabel("Date")
        ax.set_ylabel("Closing Price")
        ax.set_title(f"{ticker} Closing Prices")
        st.pyplot(fig)

        st.subheader("ARIMA Forecast")

        # ARIMA Model
        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()

        # Forecast till June 2027
        target_date = pd.Timestamp("2027-06-30")

        last_date = close_prices.index[-1]

        forecast_steps = (
            (target_date.year - last_date.year) * 12 +
            (target_date.month - last_date.month)
        )

        forecast = model_fit.forecast(steps=forecast_steps)

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_steps,
            freq='M'
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast Price": forecast
        })

        st.dataframe(forecast_df.tail(12))

        june_2027_price = forecast_df[
            forecast_df["Date"].dt.strftime("%Y-%m") == "2027-06"
        ]

        if not june_2027_price.empty:
            predicted_price = june_2027_price["Forecast Price"].iloc[0]

            st.success(
                f"Predicted Stock Price for June 2027: ${predicted_price:.2f}"
            )

        # Plot forecast

        fig2, ax2 = plt.subplots(figsize=(12,5))

        ax2.plot(
            close_prices.index,
            close_prices.values,
            label="Historical"
        )

        ax2.plot(
            forecast_df["Date"],
            forecast_df["Forecast Price"],
            label="Forecast"
        )

        ax2.set_title(f"{ticker} ARIMA Forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        st.pyplot(fig2)
