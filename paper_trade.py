import json
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# === Project imports ===
from src.data.loaders import load_ohlcv
from src.features.selected_features import make_selected_features
from src.modeling.regime_filter import volatility_regime
from src.modeling.strategy_v1 import backtest_sized

# === Config ===
ENTER, EXIT, COOLDOWN, RISK_CAP = 0.58, 0.53, 6, 0.8
REGIME_LOOKBACK, REGIME_Q = 48, 0.6

STATE_FILE = Path("paper_state.json")
TRADES_CSV = Path("paper_trades.csv")
RUN_LOG = Path("signals_log.csv")

# ---------- Helpers: NEVER break on empty/missing files ----------
def read_csv_or_empty(path, **kwargs) -> pd.DataFrame:
    """Read CSV safely; return empty DataFrame if missing or empty."""
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()
    except Exception:
        # Any parsing hiccup: treat as empty as well
        return pd.DataFrame()

def append_csv(path, row: dict):
    """
    Append one row to CSV with header control.
    Never reads the existing file (avoids EmptyDataError).
    """
    path = Path(path)
    write_header = (not path.exists()) or (path.stat().st_size == 0)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", index=False, header=write_header)

def load_state():
    try:
        if STATE_FILE.exists() and STATE_FILE.stat().st_size > 0:
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    # default state
    return {"equity": 10000.0, "position": 0.0, "last_price": None, "last_ts": None}

def save_state(state):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        # last-ditch: don't fail the run if FS burps
        pass

# ---------- Main logic (hardened) ----------
def main():
    # 1) Load data safely
    try:
        df = load_ohlcv("BTC-USD", interval="1h", period="720d")
        if df is None or df.empty or "close" not in df.columns:
            raise ValueError("No OHLCV data")
    except Exception:
        # If yfinance throttles / net issue: create a minimal placeholder to avoid crash
        now = pd.Timestamp.utcnow().floor("H")
        state = load_state()
        # Keep state, log, and exit gracefully (no trade)
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "NO_DATA", "position": state.get("position", 0.0),
            "price": state.get("last_price", 0.0), "equity": state.get("equity", 10000.0),
        })
        print(f"[NO_DATA] {now} — skipped due to missing market data.")
        return

    # 2) Features & regime (guard against weird edges)
    try:
        X_all = make_selected_features(df)
        if X_all is None or X_all.empty:
            raise ValueError("Feature matrix empty")

        px = df["close"].reindex(X_all.index).dropna()
        if px.empty:
            raise ValueError("No aligned closes")

        regime_all = volatility_regime(df, lookback=REGIME_LOOKBACK, quantile=REGIME_Q)
        regime = regime_all.reindex(X_all.index).fillna(0)
    except Exception:
        # If anything goes wrong, go flat & log
        now = pd.Timestamp.utcnow().floor("H")
        state = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "FEATURE_FAIL", "position": state.get("position", 0.0),
            "price": state.get("last_price", 0.0), "equity": state.get("equity", 10000.0),
        })
        print(f"[FEATURE_FAIL] {now} — skipped due to feature/regime error.")
        return

    # 3) Pick a recent window
    if X_all.index.size < 5:
        # not enough bars; log & stop
        now = pd.Timestamp.utcnow().floor("H")
        state = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "TOO_FEW_BARS", "position": state.get("position", 0.0),
            "price": state.get("last_price", 0.0), "equity": state.get("equity", 10000.0),
        })
        print(f"[TOO_FEW_BARS] {now} — skipped.")
        return

    try:
        cutoff = X_all.index.max() - pd.DateOffset(months=12)
        tr_mask = (X_all.index >= cutoff) & (X_all.index < X_all.index.max()) & (regime == 1)

        y_all = (px.shift(-1) > px).astype(int)

        # 4) Train only if we have enough samples
        n_tr = int(tr_mask.sum())
        if n_tr < 50:
            # Not enough samples; fallback to p_up=0.5 (neutral)
            recent_mask = X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))
            recent_idx = X_all.index[recent_mask]
            if recent_idx.empty:
                recent_idx = pd.Index([X_all.index.max()])
            p_up = pd.Series(0.5, index=recent_idx)
        else:
            pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400))
            pipe.fit(X_all.loc[tr_mask], y_all.loc[tr_mask])

            recent_mask = X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))
            recent_idx = X_all.index[recent_mask]
            if recent_idx.empty:
                recent_idx = pd.Index([X_all.index.max()])

            p_up = pd.Series(pipe.predict_proba(X_all.loc[recent_idx])[:, 1], index=recent_idx)

        # 5) Backtest last few days to size position (safe)
        try:
            pnl, eq, _ = backtest_sized(px.loc[recent_idx], p_up, ENTER, EXIT, COOLDOWN, RISK_CAP)
        except Exception:
            # If backtest fails for any reason, pretend flat
            pnl, eq = pd.Series(dtype=float), pd.Series(dtype=float)

        now = recent_idx[-1]
        current_p = float(p_up.iloc[-1])
        is_regime = int(regime.loc[now]) == 1

    except Exception:
        # Any unexpected error: be safe & flat
        now = pd.Timestamp.utcnow().floor("H")
        state = load_state()
        append_csv(RUN_LOG, {
            "datetime": now, "p_up": 0.5, "regime": 0,
            "action": "MODEL_FAIL", "position": state.get("position", 0.0),
            "price": state.get("last_price", 0.0), "equity": state.get("equity", 10000.0),
        })
        print(f"[MODEL_FAIL] {now} — skipped due to model error.")
        return

    # 6) Positioning & accounting (never crash)
    state = load_state()
    try:
        price = float(px.loc[now])
    except Exception:
        price = float(px.iloc[-1]) if not px.empty else float(state.get("last_price", 0.0))

    prev_pos = float(state.get("position", 0.0))

    if not is_regime:
        action = "FLAT_REGIME_LOW"
        target_pos = 0.0
    else:
        if current_p > ENTER:
            action = "ENTER_OR_HOLD_LONG"
            target_pos = min(RISK_CAP, max(0.0, 2 * (current_p - 0.5)))
        elif current_p < EXIT:
            action = "EXIT_TO_FLAT"
            target_pos = 0.0
        else:
            action = "HOLD"
            target_pos = prev_pos

    last_price = state.get("last_price", price) or price
    equity = float(state.get("equity", 10000.0))
    try:
        equity *= (1 + prev_pos * (price / last_price - 1))
    except Exception:
        pass

    FEE = (5 + 1) / 1e4
    pos_change = target_pos - prev_pos
    equity *= (1 - abs(pos_change) * FEE)

    state = {"equity": equity, "position": target_pos, "last_price": price, "last_ts": now.isoformat()}
    save_state(state)

    # 7) Logs (never crash)
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
                "datetime": now, "side": "BUY" if pos_change > 0 else "SELL",
                "pos_before": prev_pos, "pos_after": target_pos, "price": price, "equity_after": equity
            })
        except Exception:
            pass

    print(f"Now {now} | p_up={current_p:.3f} | regime={is_regime} | action={action} "
          f"| pos={target_pos:.3f} | price={price:.2f} | equity={equity:.2f}")


if __name__ == "__main__":
    main()
