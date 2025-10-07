import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import duckdb

from signal_model import generate_signals



# -------------------------
# 1) Create synthetic daily OHLCV
# -------------------------
np.random.seed(42)
n_days = 1000
start_price = 100.0
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)  # business days
returns = np.random.normal(loc=0.0002, scale=0.02, size=n_days)   # small drift + noise
price = start_price * np.exp(np.cumsum(returns))  # geometric random walk



# Make OHLC
high = price * (1 + np.abs(np.random.normal(0, 0.003, size=n_days)))
low = price * (1 - np.abs(np.random.normal(0, 0.003, size=n_days)))
open_ = price * (1 + np.random.normal(0, 0.001, size=n_days))
close = price.copy()
volume = np.random.randint(100, 1000, size=n_days)

df = pd.DataFrame({
    "timestamp": dates,
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "volume": volume
}).set_index("timestamp")


# connect to your DuckDB file
conn = duckdb.connect("/Users/pengdadong/Downloads/binance_merged.db")

query = """
select *
from main.ohlcv_1m
where symbol = 'DOTUSDT'
"""

df = conn.execute(query).fetchdf()
df.drop(columns=['open_time', 'close_time'], inplace=True)
#print(df)

# Set index to open_standard for resampling
daily_df = df.set_index('open_standard').resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'quote_volume': 'sum',
    'trades': 'sum'
}).reset_index()

df = daily_df.set_index('open_standard').dropna()
#print(df.head())
df = df[df.index > pd.Timestamp('2020-12-01')]

# -------------------------
# 2) Strategy: RSI signal
# -------------------------
#fast, slow = 20, 50
#df["sma_fast"] = df["close"].rolling(window=fast).mean()
#df["sma_slow"] = df["close"].rolling(window=slow).mean()

# Raw signal: 1 = long, 0 = flat
#df["signal_raw"] = (df["sma_fast"] > df["sma_slow"]).astype(int)
# Trade trigger: +1 = buy, -1 = sell
#df["signal"] = df["signal_raw"].diff().fillna(0)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# compute RSI(14)
df["rsi"] = compute_rsi(df["close"], 7)

# build {-1, 0, 1} indicator
df["signal"] = 0
df.loc[df["rsi"] < 15, "signal"] = 1     # RSI < 25 → BUY
df.loc[df["rsi"] > 85, "signal"] = -1    # RSI > 75 → SELL



# -------------------------
# 3) Backtester
# -------------------------
initial_cash = 1000000.0
cash = initial_cash
position = 0.0  # number of shares
fee_per_trade = 1.0     # flat fee (should be a portion of cost)
slippage = 0.0005       # fraction of price
trade_log = []
equity_curve = []

for idx, row in df.iterrows():
    price = row["close"]

    # Entry: Buy
    if row["signal"] == 1.0 and position * price <= .5 * cash:
        executed_price = price * (1 + slippage)
        shares = (.05 * cash) // executed_price
        if cash > 0.0:
            if shares > 0:
                cost = shares * executed_price + fee_per_trade
                cash -= cost
                position += shares
                trade_log.append({"timestamp": idx, "type": "BUY",
                              "price": executed_price, "shares": shares, "cash": cash})

    # Exit: Sell
    elif row["signal"] == -1.0 and position > 0.0:
        executed_price = price * (1 - slippage)
        proceeds = .1 * position * executed_price - fee_per_trade
        cash += proceeds
        trade_log.append({"timestamp": idx, "type": "SELL",
                          "price": executed_price, "shares": position, "cash": cash})
        position = .9 * position

    # Track portfolio value
    equity = cash + position * price
    equity_curve.append({"timestamp": idx, "equity": equity, "cash": cash,
                         "position": position, "price": price})

equity_df = pd.DataFrame(equity_curve).set_index("timestamp")
trades_df = pd.DataFrame(trade_log)
final_equity = equity_df["equity"].iloc[-1]

# -------------------------
# 4) Performance metrics
# -------------------------
equity_df["returns"] = equity_df["equity"].pct_change().fillna(0)
total_return = (final_equity / initial_cash) - 1.0

trading_days = 365
years = len(equity_df) / trading_days
annualized_return = (1 + total_return) ** (1 / max(years, 1e-9)) - 1

sharpe = (equity_df["returns"].mean() / (equity_df["returns"].std() + 1e-9)) * np.sqrt(trading_days)

cum_max = equity_df["equity"].cummax()
drawdown = (equity_df["equity"] - cum_max) / cum_max
max_drawdown = drawdown.min()

# -------------------------
# 5) Visualization
# -------------------------
out_dir = "kline_backtest_outputs"
os.makedirs(out_dir, exist_ok=True)

# Price + trade signals
plt.figure(figsize=(12,6))
plt.plot(df.index, df["close"], label="Close Price")
if not trades_df.empty:
    buys = trades_df[trades_df["type"]=="BUY"]
    sells = trades_df[trades_df["type"]=="SELL"]
    plt.scatter(buys["timestamp"], buys["price"], marker="^", s=80, label="Buy")
    plt.scatter(sells["timestamp"], sells["price"], marker="v", s=80, label="Sell")
plt.legend()
plt.title("Price and Trade Signals")
plt.savefig(os.path.join(out_dir, "DOT_price_with_signals.png"))

# Equity curve
plt.figure(figsize=(12,6))
plt.plot(equity_df.index, equity_df["equity"], label="Equity")
plt.legend()
plt.title("Equity Curve")
plt.text(0.5, 0.9, f"Sharpe Ratio: {sharpe:.2f}",
         transform=plt.gca().transAxes,   # use axes (0–1) coordinates
         fontsize=10, color='green',
         verticalalignment='top')
plt.text(0.5, 0.85, f"Total Return: {total_return:.2f}",
         transform=plt.gca().transAxes,   # use axes (0–1) coordinates
         fontsize=10, color='green',
         verticalalignment='top')
plt.text(0.5, 0.8, f"Max Drawdown: {max_drawdown:.2f}",
         transform=plt.gca().transAxes,   # use axes (0–1) coordinates
         fontsize=10, color='green',
         verticalalignment='top')
plt.savefig(os.path.join(out_dir, "DOT_equity_curve.png"))

# RSI curve
plt.figure(figsize=(12,4))
plt.plot(df.index, df["rsi"], label="RSI(7)")
plt.axhline(15, color='green', linestyle='--')
plt.axhline(85, color='red', linestyle='--')
plt.legend()
plt.title("RSI")
plt.savefig(os.path.join(out_dir, "DOT_rsi_curve.png"))

# -------------------------
# 6) Print Summary
# -------------------------
#print("Equity Curve:", equity_curve)
print("Cash:", cash)
print(df["rsi"].describe())
print(df["signal"].value_counts())
print("Initial Cash:", initial_cash)
print("Final Equity:", final_equity)
print("Total Return: {:.2%}".format(total_return))
print("Annualized Return: {:.2%}".format(annualized_return))
print("Sharpe Ratio:", round(sharpe, 2))
print("Max Drawdown: {:.2%}".format(max_drawdown))
#print("header of trades_df: ", df.head())
#print("Number of Trades:", df.tail())
