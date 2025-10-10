import json
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

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"equity": 10000.0, "position": 0.0, "last_price": None, "last_ts": None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def append_csv(path, row: dict):
    df = pd.DataFrame([row]).set_index("datetime")
    # only try to read if the file exists and has content
    if path.exists() and path.stat().st_size > 0:
        try:
            old = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
            df = pd.concat([old, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        except pd. errors.EmptyDataError:
            # file exists but has no rows/header; just start fresh
            pass
    df.to_csv(path)

def main():
    df = load_ohlcv("BTC-USD", interval="1h", period="720d")
    X_all = make_selected_features(df)
    px = df["close"].loc[X_all.index]
    regime = volatility_regime(df, lookback=REGIME_LOOKBACK, quantile=REGIME_Q).loc[X_all.index]
    mask = regime == 1

    cutoff = X_all.index.max() - pd.DateOffset(months=12)
    tr_mask = (X_all.index >= cutoff) & (X_all.index < X_all.index.max()) & mask

    y_all = (px.shift(-1) > px).astype(int)
    pipe = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=400))
    pipe.fit(X_all.loc[tr_mask], y_all.loc[tr_mask])

    recent_idx = X_all.index[X_all.index >= (X_all.index.max() - pd.Timedelta(days=3))]
    p_up = pd.Series(pipe.predict_proba(X_all.loc[recent_idx])[:,1], index=recent_idx)

    pnl, eq, _ = backtest_sized(px.loc[recent_idx], p_up, ENTER, EXIT, COOLDOWN, RISK_CAP)

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

    FEE = (5+1)/1e4
    pos_change = target_pos - prev_pos
    equity *= (1 - abs(pos_change)*FEE)

    state = {"equity": equity, "position": target_pos, "last_price": price, "last_ts": now.isoformat()}
    save_state(state)

    append_csv(RUN_LOG, {
        "datetime": now, "p_up": current_p, "regime": int(is_regime),
        "action": action, "position": target_pos, "price": price, "equity": equity
    })
    if pos_change != 0.0:
        append_csv(TRADES_CSV, {
            "datetime": now, "side": "BUY" if pos_change>0 else "SELL",
            "pos_before": prev_pos, "pos_after": target_pos, "price": price, "equity_after": equity
        })

    print(f"Now {now} | p_up={current_p:.3f} | regime={is_regime} | action={action} | pos={target_pos:.3f} | price={price:.2f} | equity={equity:.2f}")

if __name__ == "__main__":
    main()
