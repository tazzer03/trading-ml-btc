import pandas as pd
import numpy as np

def make_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low = df["close"], df["high"], df["low"]
    X = pd.DataFrame(index=df.index)

    # --- Trend strength ---
    X["sma_24"] = close.rolling(24).mean()
    X["sma_72"] = close.rolling(72).mean()
    X["sma_ratio"] = X["sma_24"] / X["sma_72"] - 1

    # --- Bollinger Band width ---
    rolling = close.rolling(24)
    X["bollinger_width"] = (rolling.mean() + 2*rolling.std() - (rolling.mean() - 2*rolling.std())) / rolling.mean()

    # --- Rolling z-score of price ---
    X["zscore_24"] = (close - rolling.mean()) / rolling.std()

    # --- Volatility clusters ---
    returns = close.pct_change()
    X["vol_6h"] = returns.rolling(6).std()
    X["vol_24h"] = returns.rolling(24).std()
    X["vol_ratio"] = X["vol_6h"] / X["vol_24h"]

    # --- Range and momentum ---
    X["range_ratio"] = (high - low) / close
    X["mom_24"] = close.pct_change(24)
    X["mom_72"] = close.pct_change(72)

    # --- Normalize to prevent leakage ---
    X = X.shift(1)
    return X.dropna()