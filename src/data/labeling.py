import pandas as pd
def label_next_up(df: pd.DataFrame) -> pd.Series:
    return (df["close"].shift(-1) > df["close"]).astype(int)