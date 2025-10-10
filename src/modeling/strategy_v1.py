import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits
from src.modeling.regime_filter import volatility_regime

FEE_BPS, SLIP_BPS, BPS = 5, 1, 1e-4

def backtest_sized(prices: pd.Series, p_up: pd.Series,
                   enter: float = 0.58, exit: float = 0.53,
                   cooldown: int = 6, risk_cap: float = 0.8):
    """
    Long-only. Hysteresis (enter>exit), cooldown, min-hold, probability sizing,
    inverse-vol scaling, next-bar execution with costs.
    """
    ret = prices.pct_change().fillna(0)

    # execution state
    state = 0.0
    cool = 0
    min_hold = 6     # bars
    held = 0
    pos = []

    for t, p in p_up.items():
        if cool > 0:
            cool -= 1

        # track holding time
        if state > 0:
            held += 1
        else:
            held = 0

        # enter/exit with hysteresis + cooldown + min-hold
        if state == 0 and p > enter and cool == 0:
            state = 1.0
            cool = cooldown
            held = 0
        elif state > 0 and p < exit and cool == 0 and held >= min_hold:
            state = 0.0
            cool = cooldown

        # probability-scaled size in [0, risk_cap]
        size_raw = (2 * (p - 0.5)) if state > 0 else 0.0
        size_raw = max(0.0, min(risk_cap, size_raw))
        pos.append(size_raw)

    # smooth tiny flips
    pos = pd.Series(pos, index=p_up.index).rolling(3).mean().fillna(0)

    # inverse-vol scaling (stabilize risk)
    vol = ret.rolling(24).std().replace(0, np.nan).bfill()  # deprecation-safe
    target_vol = vol.median()
    scale = (target_vol / vol).clip(0.25, 2.0)
    pos = (pos * scale).clip(0.0, risk_cap)

    # execute next bar + costs
    pos_exec = pos.shift(1).fillna(0)
    change = pos_exec.diff().abs().fillna(pos_exec.abs())
    costs = change * ((FEE_BPS + SLIP_BPS) * BPS)
    pnl = pos_exec * ret - costs
    eq = (1 + pnl).cumprod()
    return pnl, eq, float(change.sum())

def sharpe(r: pd.Series, annual: int = 8760) -> float:
    r = r.dropna()
    return 0.0 if r.std() == 0 else float((annual ** 0.5) * r.mean() / r.std())

def run():
    df = load_ohlcv()
    X_all = make_selected_features(df)
    y_all = label_next_up(df).loc[X_all.index]
    close_all = df["close"].loc[X_all.index]

    # trade only in high-vol regime (top 40%)
    regime = volatility_regime(df, lookback=48, quantile=0.6).loc[X_all.index]
    mask = regime == 1
    X, y, close = X_all.loc[mask], y_all.loc[mask], close_all.loc[mask]

    splits = rolling_splits(X.index, 12, 1, 24)
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400))

    pnls = []
    for tr, te in splits:
        model.fit(X.iloc[tr], y.iloc[tr])
        p = pd.Series(model.predict_proba(X.iloc[te])[:, 1], index=X.index[te])
        pnl, eq, turns = backtest_sized(
            close.iloc[te], p,
            enter=0.58, exit=0.53, cooldown=6, risk_cap=0.8
        )
        pnls.append(pnl)

    pnl_all = pd.concat(pnls).sort_index()
    eq = (1 + pnl_all).cumprod()

    print("\n=== Strategy v1.1 — Logistic + Regime + Hysteresis + Sizing (after costs) ===")
    print(f"Sharpe: {sharpe(pnl_all):.2f} | MaxDD={(eq/eq.cummax()-1).min():.1%} | "
          f"Trades≈{(pnl_all.diff().abs()>0).sum()/2:.1f}")

if __name__ == '__main__':
    run()
