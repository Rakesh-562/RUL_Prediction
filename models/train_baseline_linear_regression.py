import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import SUBSETS
from src.features import prepare, rolling_features, last_rows

for subset in SUBSETS:
    train, test, sensors = prepare(subset)
    train_features = rolling_features(train, sensors)
    test_features = rolling_features(test, sensors)

    drop_cols = ["unit", "cycle", "RUL"]
    feature_cols = [
        col for col in train_features.columns
        if col not in drop_cols
    ]

    X_train = train_features[feature_cols]
    y_train = train_features["RUL"]

    test_last_rows = last_rows(test_features)
    X_test = test_last_rows[feature_cols]
    y_test = test_last_rows["RUL"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = np.maximum(model.predict(X_test), 0)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"Linear Regression Baseline - {subset}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.3f}")
    print(f"accuracy: {100 * (1 - mae / y_test.mean()):.2f}%")
    print()
