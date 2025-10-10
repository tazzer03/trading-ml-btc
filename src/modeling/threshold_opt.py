import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits
from src.ml_baseline import backtest, sharpe

def optimize_threshold():
    df = load_ohlcv()
    X = make_selected_features(df)
    y = label_next_up(df).loc[X.index]
    close = df["close"].loc[X.index]

    splits = rolling_splits(X.index, 12, 1, 24)

    model = make_pipeline(StandardScaler(with_mean=False),
                          LogisticRegression(max_iter=200))

    thresholds = np.linspace(0.50, 0.65, 16)
    results = []

    for thresh in thresholds:
        pnl_all = []
        for tr, te in splits:
            model.fit(X.iloc[tr], y.iloc[tr])
            p = pd.Series(model.predict_proba(X.iloc[te])[:, 1], index=X.index[te])
            pnl, _, _ = backtest(close.iloc[te], p, thresh)
            pnl_all.append(pnl)
        pnl_all = pd.concat(pnl_all).sort_index()
        results.append({
            "thresh": thresh,
            "Sharpe": sharpe(pnl_all),
            "Trades": np.sign(pnl_all).diff().abs().sum() / 2
        })

    res_df = pd.DataFrame(results)
    print("\n=== Threshold Optimization ===")
    print(res_df.sort_values("Sharpe", ascending=False).head(8))
    best = res_df.loc[res_df["Sharpe"].idxmax()]
    print("\nBest threshold:", best.to_dict())

if __name__ == "__main__":
    optimize_threshold()