# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(layout="wide", page_title="Yahoo Finance Backtester")

# ------------------------
# Utilities
# ------------------------
def fetch_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data returned. Check ticker / date range / interval.")
    df = df[['Open','High','Low','Close','Close','Volume']].rename(columns={'Close':'Close'})
    df.index = pd.to_datetime(df.index)
    return df

def sma_signals(df, fast=20, slow=50):
    df = df.copy()
    df['SMA_fast'] = df['Close'].rolling(fast).mean()
    df['SMA_slow'] = df['Close'].rolling(slow).mean()
    df['signal'] = 0
    df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1
    df['signal'] = df['signal'].shift(1).fillna(0)  # act on next bar
    return df

def momentum_signals(df, lookback=90, threshold=0.0):
    df = df.copy()
    df['momentum'] = df['Close'].pct_change(periods=lookback)
    df['signal'] = 0
    df.loc[df['momentum'] > threshold, 'signal'] = 1
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def buy_and_hold_signals(df):
    df = df.copy()
    df['signal'] = 1
    df['signal'] = df['signal'].shift(1).fillna(0)
    return df

def backtest(df, init_cap=100000, fee=0.0005, slippage=0.0):
    """
    df: must contain 'Close' and 'signal' (1 = long, 0 = cash)
    fee: proportional fee on trade value (e.g., 0.0005 = 0.05%)
    slippage: proportional slippage applied to fill price
    """
    df = df.copy().dropna(subset=['Close','signal'])
    df['position'] = df['signal']  # fraction of portfolio in asset (0 or 1)
    df['price_next'] = df['Close'].shift(-1)  # price at which we get filled on next bar
    df['price_next'].fillna(df['Close'], inplace=True)

    cash = init_cap
    shares = 0.0
    equity = []
    trades = []

    for i, row in df.iterrows():
        target_pos = row['position']
        price = row['price_next'] * (1 + slippage)  # slippage applied
        current_value = shares * price + cash

        # determine desired shares to get to target_pos
        desired_value_in_asset = current_value * target_pos
        desired_shares = 0 if price == 0 else desired_value_in_asset / price
        delta_shares = desired_shares - shares

        # execute trade if needed
        if abs(delta_shares) > 1e-6:
            trade_value = abs(delta_shares) * price
            trade_cost = trade_value * fee
            # update cash & shares
            if delta_shares > 0:
                # buy
                cash -= trade_value + trade_cost
                shares += delta_shares
                trades.append({'date': i, 'side': 'buy', 'shares': delta_shares, 'price': price, 'cost': trade_cost})
            else:
                # sell
                cash += trade_value - trade_cost
                shares += delta_shares  # delta_shares negative
                trades.append({'date': i, 'side': 'sell', 'shares': -delta_shares, 'price': price, 'cost': trade_cost})

        equity_val = shares * price + cash
        equity.append({'date': i, 'equity': equity_val, 'cash': cash, 'shares': shares, 'price': price})

    equity_df = pd.DataFrame(equity).set_index('date')
    equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)

    # metrics
    start_val = equity_df['equity'].iloc[0]
    end_val = equity_df['equity'].iloc[-1]
    total_return = end_val / start_val - 1.0

    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = max(days / 365.25, 1/252)  # avoid division by tiny
    cagr = (end_val / start_val) ** (1/years) - 1

    # annualized volatility
    ann_vol = equity_df['returns'].std() * np.sqrt(252)

    # Sharpe (using zero risk-free)
    sharpe = (equity_df['returns'].mean() / (equity_df['returns'].std() + 1e-12)) * np.sqrt(252)

    # drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_dd = equity_df['drawdown'].min()

    # win rate (by trades)
    trades_df = pd.DataFrame(trades)
    win_rate = None
    if not trades_df.empty:
        # approximate P&L per round-trip (naive)
        # group buys & sells in sequence - crude but gives idea
        pnl_list = []
        pos = 0.0
        last_buy_price = None
        for t in trades:
            if t['side'] == 'buy':
                last_buy_price = t['price']
            elif t['side'] == 'sell' and last_buy_price is not None:
                pnl = (t['price'] - last_buy_price) * t['shares'] - (t['cost'])
                pnl_list.append(pnl)
                last_buy_price = None
        if pnl_list:
            win_rate = sum(1 for p in pnl_list if p > 0) / len(pnl_list)

    results = {
        'equity_curve': equity_df,
        'total_return': total_return,
        'cagr': cagr,
        'annual_volatility': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'trades': trades_df,
        'win_rate': win_rate
    }
    return results

# ------------------------
# Streamlit UI
# ------------------------
st.title("Yahoo Finance Backtester — simple & extendable")
with st.sidebar:
    st.header("Data & backtest settings")
    ticker = st.text_input("Ticker", value="AAPL")
    start = st.date_input("Start date", value=datetime(2018,1,1))
    end = st.date_input("End date", value=datetime.today().date())
    interval = st.selectbox("Interval", options=["1d","1wk","1mo"], index=0)

    st.subheader("Strategy")
    strategy = st.selectbox("Choose strategy", ["SMA Crossover","Momentum","Buy & Hold"])
    if strategy == "SMA Crossover":
        fast = st.number_input("Fast SMA", value=20, min_value=1)
        slow = st.number_input("Slow SMA", value=50, min_value=1)
    elif strategy == "Momentum":
        lookback = st.number_input("Momentum lookback (days)", value=90, min_value=1)
        threshold = st.number_input("Momentum threshold (pct)", value=0.0, format="%.4f")
    init_cap = st.number_input("Initial capital (USD)", value=100000)
    fee = st.number_input("Per trade proportional fee (e.g., 0.0005)", value=0.0005, format="%.5f")
    slippage = st.number_input("Slippage (proportional)", value=0.0, format="%.5f")
    run_bt = st.button("Run backtest")

# Run backtest
if run_bt:
    try:
        df = fetch_data(ticker, start.isoformat(), (end + pd.Timedelta(days=1)).isoformat(), interval)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if strategy == "SMA Crossover":
        df = sma_signals(df, fast=fast, slow=slow)
    elif strategy == "Momentum":
        df = momentum_signals(df, lookback=int(lookback), threshold=float(threshold))
    else:
        df = buy_and_hold_signals(df)

    results = backtest(df, init_cap=init_cap, fee=fee, slippage=slippage)

    eq = results['equity_curve']
    st.subheader("Performance summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total return", f"{results['total_return']*100:,.2f}%")
    col2.metric("CAGR", f"{results['cagr']*100:,.2f}%")
    col3.metric("Annual vol", f"{results['annual_volatility']*100:,.2f}%")
    col4.metric("Max drawdown", f"{results['max_drawdown']*100:,.2f}%")
    st.write(f"Sharpe (ann): {results['sharpe']:.2f}")
    if results['win_rate'] is not None:
        st.write(f"Win rate (approx per round-trip): {results['win_rate']:.2%}")

    # Price + signals
    st.subheader("Price & signals")
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df['_Close'], label='Close')
    if 'SMA_fast' in df.columns:
        ax.plot(df.index, df['SMA_fast'], label='SMA fast', linestyle='--', alpha=0.8)
    if 'SMA_slow' in df.columns:
        ax.plot(df.index, df['SMA_slow'], label='SMA slow', linestyle='-.', alpha=0.8)
    buys = df[(df['signal'] == 1) & (df['signal'].shift(1) == 0)]
    sells = df[(df['signal'] == 0) & (df['signal'].shift(1) == 1)]
    ax.scatter(buys.index, buys['Close'], marker='^', color='g', label='enter', zorder=3)
    ax.scatter(sells.index, sells['Close'], marker='v', color='r', label='exit', zorder=3)
    ax.legend()
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # Equity curve
    st.subheader("Equity curve")
    fig2, ax2 = plt.subplots(figsize=(12,4))
    ax2.plot(eq.index, eq['equity'], label='Equity')
    ax2.set_ylabel("Portfolio value")
    ax2.legend()
    st.pyplot(fig2)

    # Drawdown
    st.subheader("Drawdown")
    fig3, ax3 = plt.subplots(figsize=(12,3))
    ax3.plot(eq.index, eq['drawdown'], label='drawdown')
    ax3.fill_between(eq.index, eq['drawdown'], 0, where=eq['drawdown']<0, alpha=0.3)
    ax3.set_ylabel("Drawdown")
    ax3.legend()
    st.pyplot(fig3)

    # Trades
    st.subheader("Trades (last 50)")
    st.dataframe(results['trades'].sort_values('date', ascending=False).head(50).reset_index(drop=True))

    # Export equity curve/trades
    csv_eq = eq.to_csv().encode()
    st.download_button("Download equity curve CSV", csv_eq, file_name=f"{ticker}_equity_curve.csv")
    csv_tr = results['trades'].to_csv().encode()
    st.download_button("Download trades CSV", csv_tr, file_name=f"{ticker}_trades.csv")

    st.success("Backtest finished. Use code as baseline to add more signals or a regime detector.")

