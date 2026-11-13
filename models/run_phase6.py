"""Prediction intervals + maintenance policies for all four subsets.

Run from the project root:  python -m models.run_phase6

Engines are split 60 / 20 / 20:
  train       -> fit the Random Forest
  calibration -> measure its errors, to size the 90% intervals
  evaluation  -> run every engine from cycle 1 to failure and compare maintenance policies
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import conformal, maintenance as mt
from src.data import SUBSETS
from src.features import CAP, last_rows, prepare, rolling_features, split_engines

ROOT = Path(__file__).resolve().parents[1]

rows = []
for subset in SUBSETS:
    train, test, sensors = prepare(subset)
    feats = rolling_features(train, sensors)
    test_feats = last_rows(rolling_features(test, sensors))
    cols = [c for c in feats.columns if c not in ("unit", "cycle", "RUL")]

    tr, rest = split_engines(train, val_frac=0.4)
    cal, ev = split_engines(rest, val_frac=0.5)
    f_tr, f_cal, f_ev = (feats[feats.unit.isin(d.unit)] for d in (tr, cal, ev))

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(f_tr[cols], f_tr.RUL)

    # 1. intervals: size them on calibration engines, check them on the test set
    q = conformal.fit(rf.predict(f_cal[cols]), f_cal.RUL.to_numpy())
    pred = rf.predict(test_feats[cols])
    lower, upper = conformal.interval(pred, q)
    y = np.minimum(test_feats.RUL.to_numpy(), CAP)

    # 2. maintenance: every evaluation engine from cycle 1 to failure
    f_ev = f_ev.assign(lower=conformal.interval(rf.predict(f_ev[cols]), q)[0])
    lives = f_ev.groupby("unit").cycle.max().tolist()
    lowers = [g.lower.to_numpy() for _, g in f_ev.groupby("unit")]
    T = mt.best_interval(tr.groupby("unit").cycle.max().tolist())   # tuned on training engines

    policies = {
        "reactive": mt.reactive(lives),
        "fixed": mt.fixed_interval(lives, T),
        "predictive": mt.predictive(lives, lowers),
    }
    row = {
        "subset": subset,
        "RMSE": np.sqrt(np.mean((pred - y) ** 2)),
        "coverage": conformal.coverage(y, lower, upper),
        "width": np.mean(upper - lower),
        "fixed_T": T,
        "eval_engines": len(lives),
    }
    for name, res in policies.items():
        row[f"{name}_cost"] = mt.cost_rate(res)
        row[f"{name}_failures"] = sum(c == mt.C_FAIL for c, _ in res)
    row["saving_vs_fixed_%"] = 100 * (1 - row["predictive_cost"] / row["fixed_cost"])
    rows.append(row)
    print(row)

results = pd.DataFrame(rows)
results.to_csv(ROOT / "reports" / "phase6_results.csv", index=False)
print(results.round(2).to_string(index=False))
