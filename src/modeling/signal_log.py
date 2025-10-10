import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.modeling.regime_filter import volatility_regime
from src.modeling.strategy_v1 import backtest_sized

ENTER, EXIT, COOLDOWN, RISK_CAP = 0.58, 0.53, 6, 0.8

def main():
    df = load_ohlcv()                              # BTC-USD 1h
    X   = make_selected_features(df)
    px  = df["close"].loc[X.index]
    reg = volatility_regime(df, lookback=48, quantile=0.6).loc[X.index]
    mask = reg == 1

    cutoff = X.index.max() - pd.DateOffset(months=12)
    tr_mask = (X.index >= cutoff) & (X.index < X.index.max()) & mask

    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400))
    model.fit(X.loc[tr_mask], (px.shift(-1) > px).astype(int).loc[tr_mask])

    # compute recent window so sizing logic is consistent
    recent_idx = X.index[X.index >= (X.index.max() - pd.Timedelta(days=3))]
    p = pd.Series(model.predict_proba(X.loc[recent_idx])[:,1], index=recent_idx)

    pnl, eq, _ = backtest_sized(px.loc[recent_idx], p, ENTER, EXIT, COOLDOWN, RISK_CAP)
    now = recent_idx[-1]
    row = {
        "datetime": now,
        "p_up": float(p.iloc[-1]),
        "regime": int(reg.loc[now]),
        "signal": ("ENTER/HOLD LONG" if (reg.loc[now]==1 and p.iloc[-1]>ENTER)
                   else ("EXIT/HOLD FLAT" if p.iloc[-1]<EXIT or reg.loc[now]==0 else "HOLD")),
        "equity_sim": float(eq.iloc[-1]),
        "price": float(px.loc[now])
    }

    logf = "signals_log.csv"
    df_out = pd.DataFrame([row]).set_index("datetime")
    try:
        old = pd.read_csv(logf, parse_dates=["datetime"]).set_index("datetime")
        out = pd.concat([old, df_out]).sort_index().drop_duplicates()
    except FileNotFoundError:
        out = df_out
    out.to_csv(logf)
    print("Logged:", row)

if __name__ == "__main__":
    main()
