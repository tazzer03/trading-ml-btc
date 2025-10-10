import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c,h,l = df["close"], df["high"], df["low"]
    X = pd.DataFrame(index=df.index)
    X["ret_1"] = c.pct_change(1)
    X["ret_3"] = c.pct_change(3)
    X["ret_6"] = c.pct_change(6)
    X["ret_24"] = c.pct_change(24)
    X["rsi14"] = RSIIndicator(c).rsi()
    X["stoch_k"] = StochasticOscillator(h,l,c).stoch()
    X["macd_diff"] = MACD(c).macd_diff()
    X["vol_24"] = c.pct_change().rolling(24).std()
    X["vol_72"] = c.pct_change().rolling(72).std()
    X["tr_range"] = (h - l) / c
    X = X.shift(1)           # use only info known before next bar
    return X.dropna()