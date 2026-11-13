"""Phase 1 sanity check: load every C-MAPSS subset and print a summary.

Run from the project root:  python -m src.check_data
"""
from src.data import SUBSETS, load_subset, load_test_rul


def main():
    print(f"{'subset':<7}{'train rows':>11}{'train units':>12}{'test rows':>10}"
          f"{'test units':>11}{'life min/mean/max':>20}{'op. conds':>10}")
    for subset in SUBSETS:
        train, test = load_subset(subset)
        life = train.groupby("unit")["cycle"].max()
        n_conditions = train[["setting_1", "setting_2", "setting_3"]].round(0).drop_duplicates().shape[0]

        # Integrity checks
        assert train.isna().sum().sum() == 0 and test.isna().sum().sum() == 0
        assert test["unit"].nunique() == len(load_test_rul(subset))
        assert (train.groupby("unit")["RUL"].min() == 0).all()

        print(f"{subset:<7}{len(train):>11}{train['unit'].nunique():>12}{len(test):>10}"
              f"{test['unit'].nunique():>11}{f'{life.min()}/{life.mean():.0f}/{life.max()}':>20}"
              f"{n_conditions:>10}")
    print("\nAll subsets loaded and passed integrity checks.")


if __name__ == "__main__":
    main()
