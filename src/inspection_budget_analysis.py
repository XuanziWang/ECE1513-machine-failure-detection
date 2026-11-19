"""Compare trained models under fixed maintenance inspection budgets.

Run from the repository root:
    python src/inspection_budget_analysis.py

Reads existing *held-out* test scores (does not retrain models).
Saves one CSV and three figures for the ECE1513 report.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "results" / "evaluation" / "test_predictions.csv"
OUTPUT_CSV = ROOT / "results" / "evaluation" / "inspection_capacity_sensitivity.csv"
FIGURES_DIR = ROOT / "results" / "figures"

# This 5% capacity is the project's primary assumption; other budgets are
# sensitivity checks, not parameters chosen to maximize test performance.
BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.20]
MODELS = {
    "logistic_regression": ("logistic_regression_score", "Logistic Regression"),
    "rbf_svm": ("rbf_svm_score", "RBF SVM"),
    "random_forest": ("random_forest_score", "Random Forest"),
}


def load_predictions():
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_FILE}. Pull the latest main branch first."
        )
    data = pd.read_csv(PREDICTIONS_FILE)
    required = {"test_row", "actual_failure"} | {x[0] for x in MODELS.values()}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")
    if data["test_row"].duplicated().any():
        raise ValueError("Each test_row must be unique.")
    data = data.sort_values("test_row", kind="stable").reset_index(drop=True)
    if data["actual_failure"].isna().any() or not data["actual_failure"].isin([0, 1]).all():
        raise ValueError("actual_failure must contain only 0 and 1.")
    for score_col, _ in MODELS.values():
        if not np.isfinite(data[score_col].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid scores in {score_col}")
    return data


def analyze(data):
    labels = data["actual_failure"].to_numpy(dtype=int)
    total = len(labels)
    total_failures = int(labels.sum())
    if total_failures == 0:
        raise ValueError("Cannot measure failure recall without positive cases.")

    results = []
    for model_name, (score_col, _) in MODELS.items():
        scores = data[score_col].to_numpy(dtype=float)
        # Stable sorting gives a deterministic tie-break based on test_row.
        ranked_indices = np.argsort(-scores, kind="stable")
        for budget in BUDGETS:
            inspection_count = max(1, int(np.floor(total * budget)))
            flagged_indices = ranked_indices[:inspection_count]
            tp = int(labels[flagged_indices].sum())
            fp = inspection_count - tp
            fn = total_failures - tp
            precision = tp / inspection_count
            recall = tp / total_failures
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            results.append({
                "model": model_name,
                "inspection_fraction": budget,
                "inspection_percent": budget * 100,
                "inspection_count": inspection_count,
                "failures_caught": tp,
                "false_alarms": fp,
                "missed_failures": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
    return pd.DataFrame(results), total, total_failures


def save_plot(results, *, column, title, ylabel, filename, percentage=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, (_, display_name) in MODELS.items():
        subset = results.loc[results["model"] == name].sort_values("inspection_percent")
        ys = subset[column] * 100 if percentage else subset[column]
        ax.plot(subset["inspection_percent"], ys, marker="o", linewidth=2, label=display_name)
    ax.axvline(5, color="gray", linestyle="--", alpha=0.6, label="Primary budget (5%)")
    ax.set(title=title, xlabel="Inspection capacity (% of test observations)", ylabel=ylabel)
    ax.set_xticks([x * 100 for x in BUDGETS])
    ax.set_ylim(bottom=0)
    if percentage:
        ax.set_ylim(top=100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    data = load_predictions()
    results, total, total_failures = analyze(data)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)

    save_plot(results, column="recall", title="Failure recall vs. inspection capacity",
              ylabel="Recall (%)", filename="inspection_capacity_vs_recall.png", percentage=True)
    save_plot(results, column="precision", title="Inspection precision vs. capacity",
              ylabel="Precision (%)", filename="inspection_capacity_vs_precision.png", percentage=True)
    save_plot(results, column="false_alarms", title="False alarms vs. inspection capacity",
              ylabel="Number of false alarms", filename="false_alarms_vs_capacity.png")

    print(f"Test set: {total} observations, including {total_failures} actual failures")
    print("\nResults at the primary 5% inspection budget:")
    print(results.loc[results["inspection_fraction"] == 0.05,
                      ["model", "inspection_count", "failures_caught", "false_alarms", "precision", "recall"]]
          .round(4).to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved 3 figures in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
