"""Tune all three classifiers using training-only stratified cross-validation.

Run from the repository root with: python src/tune_models.py
Results and refitted pipelines are written to results/tuning/.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from joblib import parallel_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from validation import load_training_data, make_cv, make_model_pipeline


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results" / "tuning"


def main():
    X, y = load_training_data()
    folds = list(make_cv().split(X, y))
    scoring = {
        "average_precision": "average_precision",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    searches = [
        (
            "logistic_regression",
            make_model_pipeline(LogisticRegression(max_iter=5000, random_state=1513)),
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
                RandomForestClassifier(n_estimators=300, random_state=1513, n_jobs=1),
                scale=False,
            ),
            {
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 3, 5],
                "model__class_weight": [None, "balanced"],
            },
        ),
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, pipeline, parameter_grid in searches:
        print(f"\nTuning {name}", flush=True)
        search = GridSearchCV(
            pipeline,
            parameter_grid,
            scoring=scoring,
            refit="average_precision",
            cv=folds,
            n_jobs=4,
            pre_dispatch=4,
            error_score="raise",
            verbose=1,
        )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            search.fit(X, y)
        pd.DataFrame(search.cv_results_).to_csv(
            OUTPUT_DIR / f"{name}_cv_results.csv", index=False
        )
        joblib.dump(search.best_estimator_, OUTPUT_DIR / f"{name}.joblib")
        row = {"model": name, "best_parameters": search.best_params_}
        for metric in scoring:
            row[f"mean_cv_{metric}"] = float(
                search.cv_results_[f"mean_test_{metric}"][search.best_index_]
            )
            row[f"std_cv_{metric}"] = float(
                search.cv_results_[f"std_test_{metric}"][search.best_index_]
            )
        summary.append(row)
        metadata = {
            "selection_metric": "average_precision",
            "cv_folds": 5,
            "random_state": 1513,
            "training_rows": len(X),
            "training_failures": int(y.sum()),
            "sklearn_version": __import__("sklearn").__version__,
            "threshold_metrics": "Precision, recall and F1 use each estimator's default decision threshold.",
            "evaluation_scope": "Training cross-validation; scores are used for selection and need final held-out evaluation.",
            "models": summary,
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        pd.DataFrame(summary).to_csv(OUTPUT_DIR / "summary.csv", index=False)
        print(json.dumps(row, indent=2), flush=True)
    print(f"\nSaved tuning results and refitted pipelines to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
