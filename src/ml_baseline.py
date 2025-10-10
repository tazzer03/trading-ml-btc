import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.data.labeling import label_next_up
from src.modeling.walkforward import rolling_splits

FEE_BPS, SLIP_BPS, BPS = 5, 1, 1e-4

def backtest(prices, p_up, thresh=0.55):
    ret = prices.pct_change().fillna(0)
    sig = (p_up>thresh).astype(float).shift(1).fillna(0)
    change = sig.diff().abs().fillna(sig.abs())
    costs = change * ((FEE_BPS+SLIP_BPS)*BPS)
    pnl = sig*ret - costs
    equity = (1+pnl).cumprod()
    return pnl, equity, change.sum()

def sharpe(r, annual=8760):
    r = r.dropna()
    return 0.0 if r.std()==0 else float((annual**0.5)*r.mean()/r.std())

def run(model, X, y, close, splits, name):
    rows=[]
    for i,(tr,te) in enumerate(splits,1):
        m = model
        m.fit(X.iloc[tr], y.iloc[tr])
        p = pd.Series(m.predict_proba(X.iloc[te])[:,1], index=X.index[te])
        pnl, eq, turns = backtest(close.iloc[te], p)
        rows.append({"window":i, "sharpe":sharpe(pnl),
                     "max_dd": float((eq/eq.cummax()-1).min()),
                     "turnover": float(turns)})
    out = pd.DataFrame(rows)
    print(f"\n=== {name} — after costs ===")
    print(out)
    print("Aggregate: Sharpe={:.2f} | MaxDD={:.1%} | Turnover≈{:.1f}".format(
        out.sharpe.mean(), out.max_dd.mean(), out.turnover.mean()))
    return out

def main():
    df = load_ohlcv()
    X = make_selected_features(df)
    y  = label_next_up(df).loc[X.index]
    close = df["close"].loc[X.index]
    splits = rolling_splits(X.index, 12, 1, 24)

    logit = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=200))
    run(logit, X, y, close, splits, "Logistic Regression")

    lgb = LGBMClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
    run(lgb, X, y, close, splits, "LightGBM")

if __name__ == "__main__":
    main()