import pandas as pd
from src.features.indicators import make_features
from src.features.advanced_indicators import make_advanced_features

def make_selected_features(df: pd.DataFrame) -> pd.DataFrame:
    # Merge all first
    X_basic = make_features(df)
    X_adv = make_advanced_features(df)
    X = pd.concat([X_basic, X_adv], axis=1).dropna()

    # Keep only strongest features (based on your importance results)
    keep = [
        "tr_range", "stoch_k", "ret_3", "ret_1", "vol_ratio",
        "ret_6", "macd_diff", "ret_24", "vol_72", "mom_72", "zscore_24"
    ]
    X = X[keep].copy()
    return X.dropna()