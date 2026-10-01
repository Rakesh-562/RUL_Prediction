"""Loading utilities for the NASA C-MAPSS turbofan dataset."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS


def _read(path: Path) -> pd.DataFrame:
    # Files are space-separated with trailing spaces, hence the extra empty columns.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_train(subset: str = "FD001") -> pd.DataFrame:
    """Training trajectories: every engine runs until failure."""
    return _read(RAW_DIR / f"train_{subset}.txt")


def load_test(subset: str = "FD001") -> pd.DataFrame:
    """Test trajectories: each engine is cut off at some point before failure."""
    return _read(RAW_DIR / f"test_{subset}.txt")


def load_test_rul(subset: str = "FD001") -> pd.Series:
    """True RUL at the last observed cycle of each test engine, indexed by unit."""
    rul = pd.read_csv(RAW_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None).iloc[:, 0]
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return rul.rename("RUL")


def add_train_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """Add RUL = (last cycle of the engine) - (current cycle). Optionally cap it."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.assign(RUL=max_cycle - df["cycle"])
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series, cap: int | None = None) -> pd.DataFrame:
    """Add per-row RUL to test data using the true RUL at each engine's last cycle."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.assign(RUL=df["unit"].map(final_rul) + max_cycle - df["cycle"])
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def load_subset(subset: str = "FD001", cap: int | None = None):
    """Convenience: (train, test) DataFrames, both with an RUL column."""
    train = add_train_rul(load_train(subset), cap)
    test = add_test_rul(load_test(subset), load_test_rul(subset), cap)
    return train, test
