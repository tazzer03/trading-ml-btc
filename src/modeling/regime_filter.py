import pandas as pd

def volatility_regime(df: pd.DataFrame, lookback: int = 48, quantile: float = 0.6) -> pd.Series:
    """
    Regime = 1 only when recent volatility is high enough.
    - lookback: window (hours) to compute realized vol
    - quantile: trade only when vol is above this percentile (0.6 = top 40%)
    """
    returns = df["close"].pct_change()
    vol = returns.rolling(lookback).std()
    thr = vol.quantile(quantile)
    return (vol > thr).astype(int)

if __name__ == "__main__":
    from src.data.loaders import load_ohlcv
    d = load_ohlcv()
    r = volatility_regime(d, lookback=48, quantile=0.6)
    print(r.tail())
