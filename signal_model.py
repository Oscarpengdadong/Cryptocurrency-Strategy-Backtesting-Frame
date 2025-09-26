



# signal_model.py



# When moving average sma 20 > sma 50



def generate_signals(df):
    """
    df: DataFrame with OHLC data
    returns: Series of signals, 1=buy, -1=sell, 0=hold
    """
    signals = (df['close'].rolling(20).mean() > df['close'].rolling(50).mean()).astype(int)
    # Convert to -1/1
    trade_signal = signals.diff().fillna(0)
    return trade_signal