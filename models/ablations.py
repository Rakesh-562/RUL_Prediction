"""Ablation study: is each preprocessing step actually needed?

Run from the project root:  python -m models.ablations

Same Random Forest as the baseline, with one step switched off at a time:
  no_cap        - train on the true RUL instead of RUL capped at 125 (all subsets)
  no_condition  - one global normalization instead of one per operating condition (FD002, FD004)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.data import add_test_rul, add_train_rul, load_test, load_test_rul, load_train
from src.features import CAP, add_condition, last_rows, normalize, rolling_features, useful_sensors

ROOT = Path(__file__).resolve().parents[1]


def run(subset, cap=True, per_condition=True):
    train = add_train_rul(load_train(subset), cap=CAP if cap else None)
    test = add_test_rul(load_test(subset), load_test_rul(subset))
    train, test = add_condition(train, test)
    sensors = useful_sensors(train)                   # same sensors in every variant
    if not per_condition:
        train["condition"], test["condition"] = 0, 0  # one group = global normalization
    train, test = normalize(train, test, sensors)

    f_train = rolling_features(train, sensors)
    f_test = last_rows(rolling_features(test, sensors))
    cols = [c for c in f_train.columns if c not in ("unit", "cycle", "RUL")]

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(f_train[cols], f_train.RUL)
    pred = np.maximum(rf.predict(f_test[cols]), 0)
    y = f_test.RUL.to_numpy()
    return {
        "RMSE": np.sqrt(np.mean((pred - np.minimum(y, CAP)) ** 2)),
        "RMSE_uncapped": np.sqrt(np.mean((pred - y) ** 2)),
    }


rows = []
for subset in ["FD001", "FD002", "FD003", "FD004"]:
    rows.append({"subset": subset, "variant": "no_cap", **run(subset, cap=False)})
    print(rows[-1])
for subset in ["FD002", "FD004"]:
    rows.append({"subset": subset, "variant": "no_condition", **run(subset, per_condition=False)})
    print(rows[-1])

results = pd.DataFrame(rows)
results.to_csv(ROOT / "reports" / "ablation_results.csv", index=False)
print(results.round(2).to_string(index=False))
