import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits
from src.modeling.regime_filter import volatility_regime
from src.modeling.strategy_v1 import backtest_sized, sharpe

def run_report():
    df = load_ohlcv()
    X_all = make_selected_features(df)
    y_all = label_next_up(df).loc[X_all.index]
    close_all = df["close"].loc[X_all.index]

    regime = volatility_regime(df, lookback=48, quantile=0.6).loc[X_all.index]
    mask = regime == 1
    X, y, close = X_all.loc[mask], y_all.loc[mask], close_all.loc[mask]

    splits = rolling_splits(X.index, 12, 1, 24)
    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=400))

    frames = []
    for tr, te in splits:
        model.fit(X.iloc[tr], y.iloc[tr])
        p = pd.Series(model.predict_proba(X.iloc[te])[:,1], index=X.index[te])
        pnl, eq, turns = backtest_sized(close.iloc[te], p, enter=0.58, exit=0.53, cooldown=6, risk_cap=0.8)
        df_out = pd.DataFrame({
            "close": close.iloc[te],
            "p_up": p,
            "pnl": pnl,
            "equity": (1+pnl).cumprod()
        })
        frames.append(df_out)

    out = pd.concat(frames).sort_index()
    eq = out["equity"]
    s = sharpe(out["pnl"])
    mdd = (eq/eq.cummax()-1).min()

    # Save CSV
    out.to_csv("report_v1_equity_positions.csv", index_label="datetime")

    # Save chart
    plt.figure()
    eq.plot(title=f"Strategy v1.1 Equity (Sharpe {s:.2f}, MaxDD {mdd:.1%})")
    plt.xlabel("Datetime"); plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig("report_v1_equity.png")

    print("Saved: report_v1_equity.png, report_v1_equity_positions.csv")
    print(f"Sharpe {s:.2f} | MaxDD {mdd:.1%} | Bars {len(out)}")

if __name__ == "__main__":
    run_report()
