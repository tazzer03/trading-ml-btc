import json
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.modeling.regime_filter import volatility_regime
from src.modeling.strategy_v1 import backtest_sized

ENTER, EXIT, COOLDOWN, RISK_CAP = 0.58, 0.53, 6, 0.8
REGIME_LOOKBACK, REGIME_Q = 48, 0.6

STATE_FILE = Path("paper_state.json")
TRADES_CSV = Path("paper_trades.csv")
RUN_LOG = Path("signals_log.csv")

def append_csv(path, row: dict):
    path = Path(path)
    write_header = (not path.exists()) or (path.stat().st_size == 0)
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=write_header)

def load_state():
    try:
        if STATE_FILE.exists() and STATE_FILE.stat().st_size > 0:
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {"equity": 10000.0, "position": 0.0, "last_price": None, "last_ts": None}

def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass

def main():
    # Load data safely
    try:
        df = load_ohlcv("BTC-USD", interval="1h", period="720d")
        if df is None or df.empty or "close" not in df.columns:
            raise ValueError("No OHLCV data")
    except Exception:
        now = pd.Timestamp.utcnow().floor("H")
        s = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "NO_DATA", "position": s["position"],
            "price": s["last_price"] or 0.0, "equity": s["equity"],
        })
        print(f"[NO_DATA] {now}")
        return

    # Features / regime
    try:
        X_all = make_selected_features(df)
        if X_all is None or X_all.empty:
            raise ValueError("Empty features")
        px = df["close"].reindex(X_all.index).dropna()
        if px.empty:
            raise ValueError("No aligned closes")
        regime = volatility_regime(df, lookback=REGIME_LOOKBACK, quantile=REGIME_Q).reindex(X_all.index).fillna(0)
    except Exception:
        now = pd.Timestamp.utcnow().floor("H")
        s = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "FEATURE_FAIL", "position": s["position"],
            "price": s["last_price"] or 0.0, "equity": s["equity"],
        })
        print(f"[FEATURE_FAIL] {now}")
        return

    if X_all.index.size < 5:
        now = pd.Timestamp.utcnow().floor("H")
        s = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "TOO_FEW_BARS", "position": s["position"],
            "price": s["last_price"] or 0.0, "equity": s["equity"],
        })
        print(f"[TOO_FEW_BARS] {now}")
        return

    # Train or fallback neutral
    cutoff = X_all.index.max() - pd.DateOffset(months=12)
    tr_mask = (X_all.index >= cutoff) & (X_all.index < X_all.index.max()) & (regime == 1)
    y_all = (px.shift(-1) > px).astype(int)

    if int(tr_mask.sum()) < 50:
        recent_mask = X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))
        recent_idx = X_all.index[recent_mask] or pd.Index([X_all.index.max()])
        p_up = pd.Series(0.5, index=recent_idx)
    else:
        pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400))
        pipe.fit(X_all.loc[tr_mask], y_all.loc[tr_mask])
        recent_mask = X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))
        recent_idx = X_all.index[recent_mask] or pd.Index([X_all.index.max()])
        p_up = pd.Series(pipe.predict_proba(X_all.loc[recent_idx])[:, 1], index=recent_idx)

    try:
        pnl, eq, _ = backtest_sized(px.loc[recent_idx], p_up, ENTER, EXIT, COOLDOWN, RISK_CAP)
    except Exception:
        pnl, eq = pd.Series(dtype=float), pd.Series(dtype=float)

    now = recent_idx[-1]
    current_p = float(p_up.iloc[-1])
    is_regime = int(regime.loc[now]) == 1

    s = load_state()
    price = float(px.loc[now]) if now in px.index else float(px.iloc[-1])
    prev_pos = float(s["position"])

    if not is_regime:
        action, target_pos = "FLAT_REGIME_LOW", 0.0
    else:
        if current_p > ENTER:
            action, target_pos = "ENTER_OR_HOLD_LONG", min(RISK_CAP, max(0.0, 2*(current_p-0.5)))
        elif current_p < EXIT:
            action, target_pos = "EXIT_TO_FLAT", 0.0
        else:
            action, target_pos = "HOLD", prev_pos

    last_price = s["last_price"] or price
    equity = float(s["equity"])
    try:
        equity *= (1 + prev_pos * (price/last_price - 1))
    except Exception:
        pass

    FEE = (5+1)/1e4
    pos_change = target_pos - prev_pos
    equity *= (1 - abs(pos_change)*FEE)

    s = {"equity": equity, "position": target_pos, "last_price": price, "last_ts": now.isoformat()}
    save_state(s)

    try:
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": current_p, "regime": int(is_regime),
            "action": action, "position": target_pos, "price": price, "equity": equity
        })
    except Exception:
        pass

    if pos_change != 0.0:
        try:
            append_csv(TRADES_CSV, {
                "datetime": now, "side": "BUY" if pos_change>0 else "SELL",
                "pos_before": prev_pos, "pos_after": target_pos, "price": price, "equity_after": equity
            })
        except Exception:
            pass

    print(f"Now {now} | p_up={current_p:.3f} | regime={is_regime} | action={action} | "
          f"pos={target_pos:.3f} | price={price:.2f} | equity={equity:.2f}")

if __name__ == "__main__":
    main()
