from typing import List, Tuple
import numpy as np, pandas as pd

def rolling_splits(index: pd.DatetimeIndex, train_months=12, test_months=1, purge_hours=24
                  ) -> List[Tuple[np.ndarray,np.ndarray]]:
    idx = pd.Series(range(len(index)), index=index)
    splits=[]; start=index.min()
    while True:
        tr_end = start + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > index.max(): break
        tr = idx[(idx.index>=start)&(idx.index<tr_end)].values
        te = idx[(idx.index>=tr_end+pd.Timedelta(hours=purge_hours))&(idx.index<te_end)].values
        if len(tr) and len(te): splits.append((tr,te))
        start = start + pd.DateOffset(months=test_months)
    return splits