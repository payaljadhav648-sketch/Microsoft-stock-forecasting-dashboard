import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(layout="wide")
st.title("Microsoft Stock Price Forecasting Dashboard")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("final_dashboard_data.csv")

# Clean + Convert
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])

# -------------------------------
# Required Columns Check
# -------------------------------
required_cols = ['Date', 'Actual', 'LSTM_Predicted', 'ARIMA_Predicted']

if not all(col in df.columns for col in required_cols):
    st.error("Dataset format is incorrect!")
    st.write("Columns found:", df.columns)

else:
    # -------------------------------
    # Sort Data
    # -------------------------------
    df = df.sort_values('Date')

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("Filter Data")

    start_date = st.sidebar.date_input("Start Date", df['Date'].min())
    end_date = st.sidebar.date_input("End Date", df['Date'].max())

    filtered_df = df[
        (df['Date'] >= pd.to_datetime(start_date)) & 
        (df['Date'] <= pd.to_datetime(end_date))
    ].copy()

    # -------------------------------
    # Buy/Sell Signal Calculation
    # -------------------------------
    filtered_df['MA20'] = filtered_df['Actual'].rolling(window=20).mean()
    filtered_df['MA50'] = filtered_df['Actual'].rolling(window=50).mean()

    filtered_df['Signal'] = 0
    filtered_df.loc[filtered_df['MA20'] > filtered_df['MA50'], 'Signal'] = 1
    filtered_df.loc[filtered_df['MA20'] < filtered_df['MA50'], 'Signal'] = -1

    filtered_df['Position'] = filtered_df['Signal'].diff()

    # -------------------------------
    # Backtesting Strategy
    # -------------------------------
    filtered_df['Returns'] = filtered_df['Actual'].pct_change().fillna(0)
    filtered_df['Strategy_Returns'] = (
        filtered_df['Signal'].shift(1) * filtered_df['Returns']
    ).fillna(0)

    filtered_df['Cumulative_Market'] = (1 + filtered_df['Returns']).cumprod()
    filtered_df['Cumulative_Strategy'] = (1 + filtered_df['Strategy_Returns']).cumprod()

    # Clean starting point
    if len(filtered_df) > 0:
        filtered_df.iloc[0, filtered_df.columns.get_loc('Cumulative_Market')] = 1
        filtered_df.iloc[0, filtered_df.columns.get_loc('Cumulative_Strategy')] = 1

    # -------------------------------
    # KPI Metrics
    # -------------------------------
    st.subheader("Key Metrics")

    rmse_lstm = ((filtered_df['Actual'] - filtered_df['LSTM_Predicted'])**2).mean()**0.5
    rmse_arima = ((filtered_df['Actual'] - filtered_df['ARIMA_Predicted'])**2).mean()**0.5

    col1, col2 = st.columns(2)
    col1.metric("LSTM RMSE", f"{rmse_lstm:.2f}")
    col2.metric("ARIMA RMSE", f"{rmse_arima:.2f}")

    # -------------------------------
    # CAGR & Volatility
    # -------------------------------
    st.subheader("Growth & Risk Metrics")

    if len(filtered_df) > 1:
        start_price = filtered_df['Actual'].iloc[0]
        end_price = filtered_df['Actual'].iloc[-1]
        num_days = (filtered_df['Date'].iloc[-1] - filtered_df['Date'].iloc[0]).days
        years = num_days / 365 if num_days > 0 else 1

        cagr = ((end_price / start_price) ** (1 / years) - 1) * 100

        returns = filtered_df['Actual'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100

        col3, col4 = st.columns(2)
        col3.metric("CAGR (%)", f"{cagr:.2f}%")
        col4.metric("Volatility (%)", f"{volatility:.2f}%")

        st.subheader("Growth & Risk Insights")

        if cagr > 15:
            st.success("Strong long-term growth observed.")
        elif cagr > 5:
            st.info("Moderate growth observed.")
        else:
            st.warning("Low growth observed.")

        if volatility > 30:
            st.warning("High risk (volatility).")
        elif volatility > 15:
            st.info("Moderate risk.")
        else:
            st.success("Low risk (stable).")

    else:
        st.warning("Not enough data for CAGR/Volatility.")

    # -------------------------------
    # Model Comparison
    # -------------------------------
    st.subheader("Model Comparison")

    fig, ax = plt.subplots(figsize=(16,6))
    ax.plot(filtered_df['Date'], filtered_df['Actual'], label='Actual', linewidth=2)
    ax.plot(filtered_df['Date'], filtered_df['LSTM_Predicted'], label='LSTM', linestyle='--')
    ax.plot(filtered_df['Date'], filtered_df['ARIMA_Predicted'], label='ARIMA', linestyle=':')

    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.xticks(rotation=45)
    ax.legend()

    st.pyplot(fig)

    # -------------------------------
    # Model Insight
    # -------------------------------
    st.subheader("Insights")

    if rmse_lstm < rmse_arima:
        st.success(f"LSTM better (RMSE: {rmse_lstm:.2f})")
    else:
        st.success(f"ARIMA better (RMSE: {rmse_arima:.2f})")

    # -------------------------------
    # Buy/Sell Signals
    # -------------------------------
    if filtered_df[['MA20','MA50']].isna().all().any():
        st.warning("Select larger date range for signals.")
    else:
        st.subheader("Buy/Sell Signals")

        fig2, ax2 = plt.subplots(figsize=(16,6))
        ax2.plot(filtered_df['Date'], filtered_df['Actual'], label='Price')
        ax2.plot(filtered_df['Date'], filtered_df['MA20'], '--', label='MA20')
        ax2.plot(filtered_df['Date'], filtered_df['MA50'], ':', label='MA50')

        buy = filtered_df[filtered_df['Position'] == 1]
        sell = filtered_df[filtered_df['Position'] == -1]

        ax2.scatter(buy['Date'], buy['Actual'], marker='^', s=100, label='Buy')
        ax2.scatter(sell['Date'], sell['Actual'], marker='v', s=100, label='Sell')

        ax2.legend()
        plt.xticks(rotation=45)

        st.pyplot(fig2)

    # -------------------------------
    # Backtesting Plot
    # -------------------------------
    st.subheader("Strategy Backtesting")

    fig3, ax3 = plt.subplots(figsize=(16,6))
    ax3.plot(filtered_df['Date'], filtered_df['Cumulative_Market'], label='Market')
    ax3.plot(filtered_df['Date'], filtered_df['Cumulative_Strategy'], '--', label='Strategy')

    ax3.legend()
    plt.xticks(rotation=45)

    st.pyplot(fig3)

    # -------------------------------
    # Strategy Metrics
    # -------------------------------
    st.subheader("Strategy Performance Metrics")

    if len(filtered_df) > 1:
        market_ret = (filtered_df['Cumulative_Market'].iloc[-1] - 1) * 100
        strategy_ret = (filtered_df['Cumulative_Strategy'].iloc[-1] - 1) * 100

        col5, col6 = st.columns(2)
        col5.metric("Market Return (%)", f"{market_ret:.2f}%")
        col6.metric("Strategy Return (%)", f"{strategy_ret:.2f}%")

        st.subheader("Backtesting Insights")

        diff = strategy_ret - market_ret

        if diff > 0:
            st.success(f"Strategy outperformed by {diff:.2f}%")
        else:
            st.error(f"Strategy underperformed by {abs(diff):.2f}%")

    else:
        st.warning("Not enough data for backtesting.")

    # -------------------------------
    # Dataset Preview
    # -------------------------------
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.tail())
