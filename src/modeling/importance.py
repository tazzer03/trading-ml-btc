import pandas as pd
from lightgbm import LGBMClassifier
from src.data.loaders import load_ohlcv
from src.features.indicators import make_features
from src.features.advanced_indicators import make_advanced_features
from src.data.labeling import label_next_up

def feature_importance():
    df = load_ohlcv()
    X_basic = make_features(df)
    X_adv = make_advanced_features(df)
    X = pd.concat([X_basic, X_adv], axis=1).dropna()
    y = label_next_up(df).loc[X.index]

    model = LGBMClassifier(n_estimators=300, learning_rate=0.03)
    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=X.columns)
    imp = imp.sort_values(ascending=False)
    print("\n=== Feature Importance ===")
    print(imp.head(15))
    print("\nLeast Important Features:")
    print(imp.tail(10))

if __name__ == "__main__":
    feature_importance()