"""Repeat the existing model comparison across stratified train/test splits.

Run from repository root: python src/seed_stability_analysis.py

Mirrors the original hyperparameter grids and reuses existing preprocessing
pipelines and inspection-budget metric calculations. Leaves original outputs unchanged.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split

from inspection_budget_analysis import BUDGETS, MODELS, analyze
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from validation import make_model_pipeline
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "seed_analysis"
SEEDS = [42, 123, 456, 789, 1513]


def get_model_searches(seed):
    """Mirror the model configurations from tune_models.py (unchanged)."""
    return [
        (
            "logistic_regression",
            make_model_pipeline(LogisticRegression(max_iter=5000, random_state=seed)),
            {
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "rbf_svm",
            make_model_pipeline(SVC(kernel="rbf", cache_size=256)),
            {
                "model__C": [0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__class_weight": [None, "balanced"],
            },
        ),
        (
            "random_forest",
            make_model_pipeline(
                RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
                scale=False,
            ),
            {
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]


def load_full_data():
    """Use exactly the same feature exclusions as preprocessing.py."""
    df = pd.read_csv(ROOT / "data" / "ai4i2020.csv")
    X = df.drop(columns=[
        "UDI", "Product ID", "Machine failure",
        "TWF", "HDF", "PWF", "OSF", "RNF",
    ])
    y = df["Machine failure"]
    # Encode all three known Type categories consistently in every split.
    X = pd.get_dummies(X, columns=["Type"], dtype=int)
    return X, y


def get_ranking_scores(model, X_test):
    """Match the score extraction used by evaluate_models.py."""
    if hasattr(model, "predict_proba"):
        positive_column = list(model.classes_).index(1)
        return model.predict_proba(X_test)[:, positive_column]
    return model.decision_function(X_test)


def run_experiments():
    X, y = load_full_data()
    all_results = []
    scoring = {
        "average_precision": "average_precision",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        # Reuse the same five-fold stratified CV as the original experiment.
        cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=seed
            )
        folds = list(cv.split(X_train, y_train))
        predictions = pd.DataFrame({
            "test_row": np.arange(len(y_test)),
            "actual_failure": y_test.to_numpy(),
        })

        for name, pipeline, parameter_grid in get_model_searches(seed):
            print(f"Tuning {name}", flush=True)
            search = GridSearchCV(
                pipeline, parameter_grid,
                scoring=scoring, refit="average_precision",
                cv=folds, n_jobs=4, pre_dispatch=4,
                error_score="raise",
            )
            with parallel_config(backend="loky", inner_max_num_threads=1):
                search.fit(X_train, y_train)
            scores = get_ranking_scores(search.best_estimator_, X_test)
            if not np.isfinite(scores).all():
                raise ValueError(f"Invalid scores for {name}, seed {seed}")
            score_col = MODELS[name][0]
            predictions[score_col] = scores

        # Reuse the *existing* metric calculation for all five budgets.
        metrics, _, _ = analyze(predictions)
        metrics.insert(0, "seed", seed)
        all_results.append(metrics)
        print(metrics.loc[metrics["inspection_fraction"] == 0.05,
                          ["model", "failures_caught", "recall"]]
              .to_string(index=False), flush=True)

    return pd.concat(all_results, ignore_index=True)


def plot_mean_recall(summary):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, (_, display_name) in MODELS.items():
        group = summary.loc[summary["model"] == name].sort_values("inspection_percent")
        x = group["inspection_percent"].to_numpy()
        avg = group["mean_recall"].to_numpy() * 100
        std = group["std_recall"].fillna(0).to_numpy() * 100
        line, = ax.plot(x, avg, marker="o", linewidth=2, label=display_name)
        ax.fill_between(x, np.maximum(0, avg - std), np.minimum(100, avg + std),
                        color=line.get_color(), alpha=0.15)
    ax.axvline(5, color="gray", linestyle="--", alpha=0.6,
               label="Primary budget (5%)")
    ax.set(xlabel="Inspection capacity (% of test observations)",
           ylabel="Mean failure recall (%)",
           title="Failure recall across train/test splits")
    ax.set_xticks([budget * 100 for budget in BUDGETS])
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "mean_recall_vs_capacity.png", dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_experiments()
    results.to_csv(OUTPUT_DIR / "all_seed_results.csv", index=False)

    summary = results.groupby(["model", "inspection_fraction", "inspection_percent"],
                              as_index=False).agg(
        mean_recall=("recall", "mean"),
        std_recall=("recall", "std"),
        mean_precision=("precision", "mean"),
        mean_failures_caught=("failures_caught", "mean"),
        mean_false_alarms=("false_alarms", "mean"),
    )
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    plot_mean_recall(summary)

    print("\nResults at primary 5% budget (mean +/- std across seeds):")
    print(summary.loc[summary["inspection_fraction"] == 0.05,
                      ["model", "mean_recall", "std_recall", "mean_failures_caught",
                       "mean_false_alarms"]].round(4).to_string(index=False))
    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
