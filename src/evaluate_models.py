"""Evaluate tuned pipelines on the held-out test set at a fixed 5% budget.

Run from the repository root: python src/evaluate_models.py
Model fitting and parameter selection are completed by tune_models.py.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from validation import PROCESSED_DIR


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
INSPECTION_FRACTION = 0.05


def main():
    X = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Machine failure"]
    if len(X) != len(y) or set(y.unique()) != {0, 1}:
        raise ValueError("Test features and binary labels must match and include both classes.")
    tuning = json.loads((RESULTS_DIR / "tuning" / "summary.json").read_text())
    inspection_count = max(1, int(np.floor(len(X) * INSPECTION_FRACTION)))
    output_dir = RESULTS_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    default_rows = []
    budget_rows = []
    predictions = pd.DataFrame({"test_row": np.arange(len(y)), "actual_failure": y})

    for entry in tuning["models"]:
        name = entry["model"]
        model = joblib.load(RESULTS_DIR / "tuning" / f"{name}.joblib")
        default_predictions = model.predict(X)
        if hasattr(model, "predict_proba"):
            positive_column = list(model.classes_).index(1)
            scores = model.predict_proba(X)[:, positive_column]
        else:
            scores = model.decision_function(X)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite ranking scores from {name}.")

        # Break equal-score ties by original test-row order, without using labels.
        order = np.argsort(-scores, kind="stable")
        budget_predictions = np.zeros(len(y), dtype=int)
        budget_predictions[order[:inspection_count]] = 1
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_default_prediction"] = default_predictions
        predictions[f"{name}_inspection_flag"] = budget_predictions

        for prediction, rows, policy in (
            (default_predictions, default_rows, "default_threshold"),
            (budget_predictions, budget_rows, "top_5_percent"),
        ):
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            rows.append({
                "model": name,
                "policy": policy,
                "inspections": int(prediction.sum()),
                "failures_caught": int(tp),
                "false_alarms": int(fp),
                "missed_failures": int(fn),
                "true_negatives": int(tn),
                "precision": precision_score(y, prediction, zero_division=0),
                "recall": recall_score(y, prediction, zero_division=0),
                "f1": f1_score(y, prediction, zero_division=0),
                "average_precision": average_precision_score(y, scores),
            })

    default_results = pd.DataFrame(default_rows)
    budget_results = pd.DataFrame(budget_rows)
    default_results.to_csv(output_dir / "default_threshold_metrics.csv", index=False)
    budget_results.to_csv(output_dir / "inspection_budget_metrics.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metadata = {
        "test_observations": len(y),
        "test_failures": int(y.sum()),
        "inspection_fraction": INSPECTION_FRACTION,
        "inspection_count": inspection_count,
        "budget_rationale": "Illustrative 5% capacity assumption chosen before test evaluation.",
        "tie_breaking": "Original test-row order, independent of labels.",
        "score_types": "Positive-class probability for logistic regression and random forest; decision_function for SVM.",
        "threshold_policy": "Estimator defaults for threshold metrics; exactly the highest-ranked 5% for budget metrics.",
    }
    (output_dir / "evaluation_settings.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("Default decision thresholds:")
    print(default_results.to_string(index=False))
    print(f"\nFixed inspection budget: {inspection_count} of {len(y)} observations")
    print(budget_results.to_string(index=False))
    print(f"\nSaved evaluation results to {output_dir}")


if __name__ == "__main__":
    main()
