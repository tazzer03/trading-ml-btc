import json
import time
from pathlib import Path
import pandas as pd
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

# ---------- small helpers ----------
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"equity": 10000.0, "position": 0.0, "last_price": None, "last_ts": None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def append_csv(path, row: dict):
    path = Path(path)
    write_header = (not path.exists()) or (path.stat().st_size == 0)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=write_header)

def log_status(kind: str, msg: str = "", when: pd.Timestamp | None = None):
    """Write a status row so we have history even when we skip a run."""
    if when is None:
        when = pd.Timestamp.utcnow()
    append_csv(RUN_LOG, {
        "datetime": when,
        "p_up": None,
        "regime": None,
        "action": f"STATUS_{kind}",
        "position": None,
        "price": None,
        "equity": None,
        "note": msg
    })

def safe_load_ohlcv(symbol: str, interval: str, period: str, retries: int = 3, pause: int = 5):
    """Retry wrapper. Returns df or None."""
    last_err = None
    for _ in range(retries):
        try:
            df = load_ohlcv(symbol, interval=interval, period=period)
            if df is not None and len(df) >= 300 and {"open","high","low","close"}.issubset(df.columns):
                return df
            last_err = Exception("DataFrame empty or missing OHLC columns")
        except Exception as e:
            last_err = e
        time.sleep(pause)
    return None

# ---------- main flow ----------
def main():
    # 1) Load data robustly
    df = safe_load_ohlcv("BTC-USD", interval="1h", period="720d")
    if df is None or df.empty:
        log_status("NO_DATA", "Could not fetch usable OHLCV after retries.")
        print("WARN: NO_DATA — skipping run gracefully.")
        return 0  # IMPORTANT: don't fail the workflow

    # 2) Build features & regime (guard lengths)
    try:
        X_all = make_selected_features(df)
        if X_all is None or X_all.empty:
            log_status("FEATURE_FAIL", "Feature builder returned empty.")
            print("WARN: FEATURE_FAIL — skipping run.")
            return 0
        px = df["close"].loc[X_all.index]
        regime = volatility_regime(df, lookback=REGIME_LOOKBACK, quantile=REGIME_Q).loc[X_all.index]
        mask = regime == 1
    except Exception as e:
        log_status("FEATURE_FAIL", f"{type(e).__name__}: {e}")
        print(f"WARN: FEATURE_FAIL — {e}")
        return 0

    if len(X_all) < 200:
        log_status("TOO_FEW_BARS", f"Only {len(X_all)} usable rows.")
        print("WARN: TOO_FEW_BARS — skipping run.")
        return 0

    # 3) Train simple model on last 12 months within regime
    try:
        cutoff = X_all.index.max() - pd.DateOffset(months=12)
        tr_mask = (X_all.index >= cutoff) & (X_all.index < X_all.index.max()) & mask
        if tr_mask.sum() < 100:
            log_status("TRAIN_SKIP", f"Training rows too few: {int(tr_mask.sum())}")
            print("WARN: TRAIN_SKIP — not enough training rows.")
            return 0

        y_all = (px.shift(-1) > px).astype(int)
        pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400))
        pipe.fit(X_all.loc[tr_mask], y_all.loc[tr_mask])

        recent_idx = X_all.index[X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))]
        if len(recent_idx) < 5:
            log_status("RECENT_TOO_SHORT", f"recent_idx={len(recent_idx)}")
            print("WARN: RECENT_TOO_SHORT — skipping.")
            return 0

        p_up = pd.Series(pipe.predict_proba(X_all.loc[recent_idx])[:, 1], index=recent_idx)
    except Exception as e:
        log_status("MODEL_FAIL", f"{type(e).__name__}: {e}")
        print(f"WARN: MODEL_FAIL — {e}")
        return 0

    # 4) Backtest execution sizing on recent window
    try:
        pnl, eq, _ = backtest_sized(px.loc[recent_idx], p_up, ENTER, EXIT, COOLDOWN, RISK_CAP)
    except Exception as e:
        log_status("BACKTEST_FAIL", f"{type(e).__name__}: {e}")
        print(f"WARN: BACKTEST_FAIL — {e}")
        return 0

    # 5) Decide action & update state
    now = recent_idx[-1]
    current_p = float(p_up.iloc[-1])
    is_regime = int(regime.loc[now]) == 1

    state = load_state()
    price = float(px.loc[now])
    prev_pos = float(state["position"])

    if not is_regime:
        action = "FLAT_REGIME_LOW"
        target_pos = 0.0
    else:
        if current_p > ENTER:
            action = "ENTER_OR_HOLD_LONG"
            target_pos = min(RISK_CAP, max(0.0, 2*(current_p-0.5)))
        elif current_p < EXIT:
            action = "EXIT_TO_FLAT"
            target_pos = 0.0
        else:
            action = "HOLD"
            target_pos = prev_pos

    last_price = state["last_price"] or price
    equity = float(state["equity"])
    equity *= (1 + prev_pos * (price/last_price - 1))

    FEE = (5 + 1) / 1e4
    pos_change = target_pos - prev_pos
    equity *= (1 - abs(pos_change) * FEE)

    state = {"equity": equity, "position": target_pos, "last_price": price, "last_ts": now.isoformat()}
    save_state(state)

    append_csv(RUN_LOG, {
        "datetime": now, "p_up": current_p, "regime": int(is_regime),
        "action": action, "position": target_pos, "price": price, "equity": equity
    })
    if pos_change != 0.0:
        append_csv(TRADES_CSV, {
            "datetime": now, "side": "BUY" if pos_change > 0 else "SELL",
            "pos_before": prev_pos, "pos_after": target_pos, "price": price, "equity_after": equity
        })

    print(f"Now {now} | p_up={current_p:.3f} | regime={is_regime} | action={action} | "
          f"pos={target_pos:.3f} | price={price:.2f} | equity={equity:.2f}")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        # final catch-all for absolute safety
        log_status("UNCAUGHT", f"{type(e).__name__}: {e}")
        print(f"WARN: UNCAUGHT — {e}")
        raise SystemExit(0)
