"""Preprocessing for C-MAPSS: RUL cap, condition-wise normalization, windows and rolling features."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data import SENSOR_COLS, SETTING_COLS, load_train, load_test, load_test_rul, add_train_rul, add_test_rul

CAP = 125     # RUL above this is treated as "healthy" (see EDA Q5)
WINDOW = 30   # cycles per window (see EDA Q9)


def useful_sensors(train):
    """Sensors with more than 3 distinct values inside every operating condition.

    Some FD002/FD004 sensors (e.g. inlet temperature) only change with the condition,
    so they are constant once we look inside one condition and carry no wear information.
    """
    n = train.groupby("condition")[SENSOR_COLS].nunique().min()
    return list(n[n > 3].index)


def add_condition(train, test):
    """Label each row with its operating condition (0 for single-condition subsets)."""
    n = len(train[SETTING_COLS].round(0).drop_duplicates())
    if n == 1:
        train["condition"], test["condition"] = 0, 0
        return train, test
    km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(train[SETTING_COLS])
    train["condition"] = km.predict(train[SETTING_COLS])
    test["condition"] = km.predict(test[SETTING_COLS])
    return train, test


def normalize(train, test, sensors):
    """Z-score each sensor within each operating condition, using training statistics only."""
    stats = train.groupby("condition")[sensors].agg(["mean", "std"])
    for c in sensors:
        mean = train["condition"].map(stats[(c, "mean")])
        std = train["condition"].map(stats[(c, "std")])
        train[c] = (train[c] - mean) / std
        mean = test["condition"].map(stats[(c, "mean")])
        std = test["condition"].map(stats[(c, "std")])
        test[c] = (test[c] - mean) / std
    return train, test


def prepare(subset="FD001"):
    """Load a subset and apply all preprocessing. Returns (train, test, sensors).

    Train RUL is capped at CAP. Test RUL is left uncapped so evaluation uses the true values.
    """
    train = add_train_rul(load_train(subset), cap=CAP)
    test = add_test_rul(load_test(subset), load_test_rul(subset))
    train, test = add_condition(train, test)
    sensors = useful_sensors(train)
    train, test = normalize(train, test, sensors)
    return train, test, sensors


def split_engines(train, val_frac=0.2, seed=0):
    """Split by engine id so no engine appears in both train and validation."""
    units = train["unit"].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=int(len(units) * val_frac), replace=False)
    val = train["unit"].isin(val_units)
    return train[~val].copy(), train[val].copy()


# ---------- deep models: sliding windows ----------

def _pad(x, window):
    """Repeat the first row if an engine has fewer than `window` cycles."""
    if len(x) >= window:
        return x
    return np.vstack([np.repeat(x[:1], window - len(x), axis=0), x])


def make_windows(df, sensors, window=WINDOW):
    """All windows of every engine. X: (n, window, n_sensors), y: RUL at the window's last cycle."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        x, rul = _pad(g[sensors].to_numpy(), window), g["RUL"].to_numpy()
        for end in range(window, len(x) + 1):
            X.append(x[end - window:end])
            y.append(rul[min(end, len(rul)) - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def last_windows(df, sensors, window=WINDOW):
    """Only the last window of each engine (how the test set is evaluated)."""
    X, y = [], []
    for _, g in df.groupby("unit"):
        X.append(_pad(g[sensors].to_numpy(), window)[-window:])
        y.append(g["RUL"].iloc[-1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------- classical models: rolling features ----------

def rolling_features(df, sensors, window=WINDOW):
    """Per engine: rolling mean, std and slope over `window` cycles, plus exponential smoothing."""
    out = df[["unit", "cycle", "RUL"]].copy()
    t = df["cycle"].astype(float)
    for c in sensors:
        g = df.groupby("unit")[c]
        roll = lambda s: s.rolling(window, min_periods=1)
        out[c + "_mean"] = g.transform(lambda s: roll(s).mean())
        out[c + "_std"] = g.transform(lambda s: roll(s).std()).fillna(0)
        out[c + "_ewm"] = g.transform(lambda s: s.ewm(alpha=0.1).mean())
        # slope of a least-squares line over the window: cov(t, x) / var(t)
        tx = (t * df[c]).groupby(df["unit"]).transform(lambda s: roll(s).mean())
        tm = t.groupby(df["unit"]).transform(lambda s: roll(s).mean())
        tt = (t * t).groupby(df["unit"]).transform(lambda s: roll(s).mean())
        var = tt - tm ** 2
        out[c + "_slope"] = ((tx - tm * out[c + "_mean"]) / var.where(var > 0)).fillna(0)
    return out


def last_rows(df):
    """Last cycle of each engine (one test prediction per engine)."""
    return df.groupby("unit").tail(1)
