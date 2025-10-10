import pandas as pd
import lightgbm as lgb
from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits
from src.ml_baseline import backtest, sharpe

def run_tuned_lgbm():
    df = load_ohlcv()
    X = make_selected_features(df)
    y = label_next_up(df).loc[X.index]
    close = df["close"].loc[X.index]

    splits = rolling_splits(X.index, 12, 1, 24)

    params = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=16,              # smaller = smoother
        max_depth=4,                # prevent overfit
        subsample=0.7,              # use only 70% of data per tree
        colsample_bytree=0.8,       # use subset of features
        reg_alpha=0.2,              # L1 regularization
        reg_lambda=0.8,             # L2 regularization
        random_state=42
    )

    results = []
    for tr, te in splits:
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = pd.Series(model.predict_proba(X.iloc[te])[:, 1], index=X.index[te])
        pnl, _, _ = backtest(close.iloc[te], p, 0.54)
        results.append(pnl)

    pnl_all = pd.concat(results).sort_index()
    print("\n=== Tuned LightGBM — after costs ===")
    print(f"Sharpe: {sharpe(pnl_all):.2f} | MaxDD={pnl_all.min():.1%} | Trades≈{(pnl_all.diff().abs()>0).sum()/2:.1f}")

if __name__ == "__main__":
    run_tuned_lgbm()