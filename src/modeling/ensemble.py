import pandas as pd
from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits
from src.modeling.regime_filter import volatility_regime
from src.ml_baseline import backtest, sharpe
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

def run_ensemble_with_regime():
    df = load_ohlcv()
    X = make_selected_features(df)
    y = label_next_up(df).loc[X.index]
    close = df["close"].loc[X.index]

    # Apply the volatility regime filter
    regime = volatility_regime(df).loc[X.index]
    active_idx = regime[regime == 1].index  # trade only when regime == 1
    X = X.loc[active_idx]
    y = y.loc[active_idx]
    close = close.loc[active_idx]

    splits = rolling_splits(X.index, 12, 1, 24)

    log_params = dict(max_iter=500)
    lgb_params = dict(
        n_estimators=250, learning_rate=0.03, num_leaves=16, max_depth=4,
        subsample=0.7, colsample_bytree=0.8, reg_lambda=0.8, random_state=42
    )

    pnl_list = []

    for tr, te in splits:
        # Train models
        log_model = LogisticRegression(**log_params).fit(X.iloc[tr], y.iloc[tr])
        lgb_model = lgb.LGBMClassifier(**lgb_params).fit(X.iloc[tr], y.iloc[tr])

        # Combine predictions
        p_log = log_model.predict_proba(X.iloc[te])[:, 1]
        p_lgb = lgb_model.predict_proba(X.iloc[te])[:, 1]
        p_combined = 0.6 * p_log + 0.4 * p_lgb  # weighted ensemble

        pnl, _, _ = backtest(close.iloc[te], pd.Series(p_combined, index=X.index[te]), 0.54)
        pnl_list.append(pnl)

    pnl_all = pd.concat(pnl_list).sort_index()

    print("\n=== Regime-Filtered Ensemble (60% Logistic + 40% LightGBM) ===")
    print(f"Sharpe: {sharpe(pnl_all):.2f} | MaxDD={pnl_all.min():.1%} | Trades≈{(pnl_all.diff().abs()>0).sum()/2:.1f}")

if __name__ == "__main__":
    run_ensemble_with_regime()

