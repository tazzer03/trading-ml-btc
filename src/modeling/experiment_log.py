import pandas as pd
from datetime import datetime
from pathlib import Path

def log_experiment(model_name, enter, exit, quantile, sharpe, maxdd, trades, notes=""):
    log_file = Path("experiment_log.csv")

    row = {
        "date_run": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "enter": enter,
        "exit": exit,
        "quantile": quantile,
        "sharpe": round(sharpe, 3),
        "maxdd": round(maxdd, 3),
        "trades": round(trades, 1),
        "notes": notes
    }

    df = pd.DataFrame([row])

    if log_file.exists():
        old = pd.read_csv(log_file)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(log_file, index=False)
    print(f"✅ Logged results for {model_name} at {row['date_run']}")

# Example of usage:
# log_experiment("logistic_v1.1", 0.56, 0.52, 0.6, 1.78, -0.8, 111, "baseline after costs")
