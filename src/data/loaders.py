import pandas as pd
import yfinance as yf

_OHLCV_NAMES = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

def _extract_ohlcv_name(col):
    """
    From a column label (which can be a str or tuple), return the OHLCV name
    ('Open','High','Low','Close','Adj Close','Volume') if present, else None.
    """
    if isinstance(col, tuple):
        for part in col:
            if isinstance(part, str) and part in _OHLCV_NAMES:
                return part
        return None
    if isinstance(col, str):
        return col if col in _OHLCV_NAMES else None
    return None

def load_ohlcv(symbol: str = "BTC-USD", interval: str = "1h", period: str = "720d") -> pd.DataFrame:
    """
    Robust OHLCV loader for yfinance that handles:
    - MultiIndex columns like ('Open','BTC-USD') or ('BTC-USD','Open')
    - Missing 'Adj Close' (falls back to 'Close')
    - Timezone-naive indices (localizes to UTC)

    Returns columns: ['open','high','low','close','adj_close','volume'] with UTC index.
    """
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=True,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol} {interval} {period}")

    raw = df.copy()

    # Map each column to its OHLCV role
    role_map = {}
    for c in raw.columns:
        role = _extract_ohlcv_name(c)
        if role is not None:
            role_map[role] = c

    # yfinance can omit 'Adj Close' for some crypto; fall back to Close
    required = {
        "Open": None, "High": None, "Low": None, "Close": None, "Adj Close": None, "Volume": None
    }
    required.update({k: role_map.get(k) for k in required})

    if required["Adj Close"] is None:
        required["Adj Close"] = required["Close"]

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not find columns {missing} in yfinance output: {list(raw.columns)}")

    out = pd.DataFrame({
        "open":  raw[required["Open"]].astype(float),
        "high":  raw[required["High"]].astype(float),
        "low":   raw[required["Low"]].astype(float),
        "close": raw[required["Close"]].astype(float),
        "adj_close": raw[required["Adj Close"]].astype(float),
        "volume": raw[required["Volume"]].astype(float),
    })

    # De-dupe/sort and make tz-aware UTC
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    return out

