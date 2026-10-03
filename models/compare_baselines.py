from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import SUBSETS
from src.features import last_rows, prepare, rolling_features

MODEL_FACTORIES = {
    "Linear Regression": LinearRegression,
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}
MODEL_COLORS = {
    "Linear Regression": "#147D92",
    "Random Forest": "#E07A32",
}
FIGURES_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def evaluate_baselines():
    results = []
    predictions_by_subset = {}

    for subset in SUBSETS:
        train, test, sensors = prepare(subset)
        train_features = rolling_features(train, sensors)
        test_features = rolling_features(test, sensors)

        feature_cols = [
            column for column in train_features.columns
            if column not in {"unit", "cycle", "RUL"}
        ]
        X_train = train_features[feature_cols]
        y_train = train_features["RUL"]

        test_last_rows = last_rows(test_features)
        X_test = test_last_rows[feature_cols]
        y_test = test_last_rows["RUL"].to_numpy()
        predictions_by_subset[subset] = {"actual": y_test}

        for model_name, make_model in MODEL_FACTORIES.items():
            model = make_model()
            model.fit(X_train, y_train)
            predictions = np.maximum(model.predict(X_test), 0)
            predictions_by_subset[subset][model_name] = predictions
            results.append({
                "subset": subset,
                "model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
            })

    return pd.DataFrame(results), predictions_by_subset


def plot_predictions(predictions_by_subset):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False, sharey=False)

    for ax, subset in zip(axes.flat, SUBSETS):
        subset_predictions = predictions_by_subset[subset]
        actual = subset_predictions["actual"]
        order = np.argsort(actual)
        engine_order = np.arange(1, len(actual) + 1)
        ax.plot(engine_order, actual[order], color="#313A40", linewidth=2.2,
                label="Actual RUL")

        for model_name in MODEL_FACTORIES:
            ax.plot(
                engine_order,
                subset_predictions[model_name][order],
                color=MODEL_COLORS[model_name],
                linewidth=1.5,
                alpha=0.9,
                label=model_name,
            )

        ax.set_title(subset)
        ax.set_xlabel("Test engines (sorted by actual RUL)")
        ax.set_ylabel("Remaining useful life")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Actual and Predicted RUL by Test Engine", y=0.99, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    output_path = FIGURES_DIR / "baseline_actual_vs_predicted.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_metrics(results):
    metrics = ["MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
    subsets = list(SUBSETS)
    positions = np.arange(len(subsets))
    bar_width = 0.34

    for ax, metric in zip(axes, metrics):
        for model_index, model_name in enumerate(MODEL_FACTORIES):
            values = (
                results[results["model"] == model_name]
                .set_index("subset")
                .loc[subsets, metric]
                .to_numpy()
            )
            offset = (model_index - 0.5) * bar_width
            bars = ax.bar(
                positions + offset,
                values,
                width=bar_width,
                color=MODEL_COLORS[model_name],
                label=model_name,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

        ax.set_title(metric)
        ax.set_xticks(positions, subsets)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if metric == "R2":
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Lower is better")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Baseline Model Metrics by CMAPSS Subset", y=0.99, fontsize=15)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_path = FIGURES_DIR / "baseline_metric_comparison.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results, predictions_by_subset = evaluate_baselines()
    print(results.to_string(index=False, formatters={
        "MAE": "{:.3f}".format,
        "RMSE": "{:.3f}".format,
        "R2": "{:.3f}".format,
    }))
    print(f"\nSaved plots:\n- {plot_predictions(predictions_by_subset)}\n- {plot_metrics(results)}")


if __name__ == "__main__":
    main()
