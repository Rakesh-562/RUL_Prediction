"""Train CNN, LSTM and GRU on all four subsets and save the test results.

Run from the project root:  python -m models.train_deep
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.compare_baselines import nasa_score
from src.data import SUBSETS
from src.deep_models import make_model, predict, train
from src.features import CAP, last_windows, make_windows, prepare, split_engines

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["CNN", "LSTM", "GRU"]

results = []
for subset in SUBSETS:
    train_df, test_df, sensors = prepare(subset)
    tr, va = split_engines(train_df)              # 80% engines for training, 20% for early stopping

    X_train, y_train = make_windows(tr, sensors)
    X_val, y_val = make_windows(va, sensors)
    X_test, y_test = last_windows(test_df, sensors)
    y_capped = np.minimum(y_test, CAP)

    for name in MODELS:
        torch.manual_seed(0)
        model = train(make_model(name, len(sensors)), X_train, y_train, X_val, y_val)
        torch.save(model.state_dict(), ROOT / "models" / f"{name}_{subset}.pt")

        pred = predict(model, X_test)
        results.append({
            "subset": subset,
            "model": name,
            "RMSE": np.sqrt(np.mean((pred - y_capped) ** 2)),
            "Score": nasa_score(y_capped, pred),
            "RMSE_uncapped": np.sqrt(np.mean((pred - y_test) ** 2)),
        })
        print(results[-1])

results = pd.DataFrame(results)
results.to_csv(ROOT / "reports" / "deep_results.csv", index=False)
print(results.round(2).to_string(index=False))
