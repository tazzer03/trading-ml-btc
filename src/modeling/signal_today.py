import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.modeling.regime_filter import volatility_regime
from src.modeling.strategy_v1 import backtest_sized

def main():
    df = load_ohlcv()
    X = make_selected_features(df)
    close = df["close"].loc[X.index]
    regime = volatility_regime(df, lookback=48, quantile=0.6).loc[X.index]
    mask = regime == 1

    # last 12 months for training
    cutoff = X.index.max() - pd.DateOffset(months=12)
    tr_mask = (X.index >= cutoff) & (X.index < X.index.max()) & mask
    te_index = X.index[-1:]  # last bar

    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400))
    model.fit(X.loc[tr_mask], (close.shift(-1) > close).astype(int).loc[tr_mask])

    p = pd.Series(model.predict_proba(X.loc[te_index])[:,1], index=te_index)

    # run sizing on the last ~3 days to compute current position consistently
    recent_idx = X.index[X.index >= (X.index.max() - pd.Timedelta(days=3))]
    p_recent = pd.Series(model.predict_proba(X.loc[recent_idx])[:,1], index=recent_idx)
    pnl, eq, _ = backtest_sized(close.loc[recent_idx], p_recent, enter=0.58, exit=0.53, cooldown=6, risk_cap=0.8)

    current_time = recent_idx[-1]
    print(f"Now: {current_time}  |  p_up={p_recent.iloc[-1]:.3f}")
    print("Paper-trade action:")
    if regime.loc[current_time] == 0:
        print(" - Regime = LOW VOL → STAY FLAT (0)")
    else:
        print(" - Regime = HIGH VOL")
        if p_recent.iloc[-1] > 0.58:
            print(" - ENTER / HOLD LONG (size increases with p_up)")
        elif p_recent.iloc[-1] < 0.53:
            print(" - EXIT to FLAT")
        else:
            print(" - HOLD previous state")

if __name__ == "__main__":
    main()
