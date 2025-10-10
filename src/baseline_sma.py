import numpy as np
import pandas as pd
from src.data.loaders import load_ohlcv

FEE_BPS = 5     # 0.05% per trade side
SLIP_BPS = 1    # 0.01% per trade side
BPS = 1e-4

def metrics(returns: pd.Series, equity: pd.Series, bars_per_year: int = 8760):
    r = returns.dropna()
    sharpe = 0.0 if r.std() == 0 else np.sqrt(bars_per_year) * r.mean() / r.std()
    peak = equity.cummax()
    mdd = ((equity / peak) - 1.0).min()
    turnover = ( ( (equity*0)+1 ).index, )  # placeholder to keep structure
    return float(sharpe), float(mdd)

def main():
    df = load_ohlcv()  # 1h BTC, ~720 days
    close = df["close"]

    # Indicators (lagged so we act next bar)
    sma_fast = close.rolling(24).mean()
    sma_slow = close.rolling(72).mean()
    signal = (sma_fast > sma_slow).astype(float).shift(1).fillna(0)  # long/flat

    # Bar returns and position
    ret = close.pct_change().fillna(0)
    pos = signal  # already shifted

    # Costs on position changes
    change = pos.diff().abs().fillna(pos.abs())
    per_side_cost = (FEE_BPS + SLIP_BPS) * BPS
    costs = change * per_side_cost

    pnl = pos * ret - costs
    equity = (1 + pnl).cumprod()

    # Metrics
    sharpe, mdd = metrics(pnl, equity)
    trades = int(change.sum())  # approx number of entries/exits
    turnover = float(change.sum())
    print("\n=== SMA(24/72) Long/Flat — After Costs ===")
    print(f"Bars: {len(df)}  |  Trades≈ {trades}")
    print(f"Sharpe: {sharpe:.2f}  |  Max Drawdown: {mdd:.1%}  |  Turnover: {turnover:.1f}")

    # Optional: plot equity (will pop a window)
    try:
        import matplotlib.pyplot as plt
        equity.plot(title="Equity Curve — SMA(24/72) BTC-USD (1h, after costs)")
        plt.show()
    except Exception as e:
        print("(Plot skipped)", e)

if __name__ == "__main__":
    main()