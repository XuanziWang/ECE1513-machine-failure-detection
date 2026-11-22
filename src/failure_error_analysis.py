"""Descriptive failure/error analysis at the existing 5% inspection budget.

ECE1513 machine-failure-detection project.
Run from the repository root:
    python src/failure_error_analysis.py

Inputs are EXISTING held-out predictions, matching test features and labels.
No model training, hyperparameter tuning, or threshold optimization occurs.
This describes one synthetic benchmark test split, not deployment performance.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_FILE = ROOT / "results" / "evaluation" / "test_predictions.csv"
FEATURES_FILE = ROOT / "data" / "processed" / "X_test.csv"
LABELS_FILE = ROOT / "data" / "processed" / "y_test.csv"
EXISTING_METRICS_FILE = ROOT / "results" / "evaluation" / "inspection_budget_metrics.csv"
OUTPUT_DIR = ROOT / "results" / "evaluation"
FIGURES_DIR = ROOT / "results" / "figures"

BUDGET_FRACTION = 0.05
MODELS = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF SVM",
    "random_forest": "Random Forest",
}
MODEL_KEYS = list(MODELS)
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TYPE_COLUMNS = ["Type_H", "Type_L", "Type_M"]
TYPE_ORDER = ["L", "M", "H"]
PATTERN_NAMES = {
    "111": "Detected by all three",
    "011": "SVM + RF only",
    "100": "Logistic Regression only",
    "110": "Logistic Regression + SVM only",
    "101": "Logistic Regression + RF only",
    "010": "SVM only",
    "001": "Random Forest only",
    "000": "Missed by all three",
}


def read_inputs():
    for path in (PREDICTIONS_FILE, FEATURES_FILE, LABELS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run from the project folder and pull the latest main branch."
            )
    pred = pd.read_csv(PREDICTIONS_FILE)
    features = pd.read_csv(FEATURES_FILE)
    labels = pd.read_csv(LABELS_FILE)
    required_prediction_columns = {"test_row", "actual_failure"}
    for model in MODELS:
        required_prediction_columns |= {
            f"{model}_inspection_flag", f"{model}_score"
        }
    missing = required_prediction_columns - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    missing = set(NUMERIC_FEATURES + TYPE_COLUMNS) - set(features.columns)
    if missing:
        raise ValueError(f"Feature file missing columns: {sorted(missing)}")
    if "Machine failure" not in labels:
        raise ValueError("Expected 'Machine failure' column in y_test.csv")
    if len(pred) != len(features) or len(pred) != len(labels):
        raise ValueError("Predictions, X_test and y_test must have identical row counts.")

    # The preprocessing/evaluation scripts write all three files in test-row order.
    # Reject mismatches rather than silently joining the wrong sensor record to a label.
    ids = pd.to_numeric(pred["test_row"], errors="raise").to_numpy()
    if not np.array_equal(ids, np.arange(len(pred))):
        raise ValueError("test_row must be the original sequential index 0..n-1")
    y = pd.to_numeric(pred["actual_failure"], errors="raise")
    external_y = pd.to_numeric(labels["Machine failure"], errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("actual_failure must be binary 0/1, without missing values")
    if not np.array_equal(y.to_numpy(dtype=int), external_y.to_numpy(dtype=int)):
        raise ValueError("Actual labels differ from y_test.csv (possible row misalignment).")
    if y.sum() == 0:
        raise ValueError("Cannot analyze missed failures when no positive labels exist.")

    one_hot = features[TYPE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if one_hot.isna().any().any() or not one_hot.isin([0, 1]).all().all():
        raise ValueError("Machine type one-hot columns must contain only 0 and 1")
    if not (one_hot.sum(axis=1) == 1).all():
        raise ValueError("Each test record must have exactly one machine Type")
    for col in NUMERIC_FEATURES:
        numeric = pd.to_numeric(features[col], errors="raise")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Non-finite sensor measurement: {col}")
        features[col] = numeric

    expected_count = max(1, int(np.floor(len(pred) * BUDGET_FRACTION)))
    for model in MODELS:
        flag_name = f"{model}_inspection_flag"
        score_name = f"{model}_score"
        flag = pd.to_numeric(pred[flag_name], errors="raise")
        score = pd.to_numeric(pred[score_name], errors="raise")
        if flag.isna().any() or not flag.isin([0, 1]).all():
            raise ValueError(f"{flag_name} must contain only 0 and 1")
        if not np.isfinite(score.to_numpy()).all():
            raise ValueError(f"{score_name} contains invalid values")
        if int(flag.sum()) != expected_count:
            raise ValueError(
                f"{model} flagged {int(flag.sum())}, expected {expected_count} "
                f"for a {BUDGET_FRACTION:.0%} budget"
            )
        # The repository's evaluation uses stable tie-breaking by test_row.
        order = np.argsort(-score.to_numpy(), kind="stable")
        expected_flags = np.zeros(len(pred), dtype=int)
        expected_flags[order[:expected_count]] = 1
        if not np.array_equal(expected_flags, flag.to_numpy(dtype=int)):
            raise ValueError(f"{model} flags do not match the top {expected_count} scores")
        pred[flag_name] = flag.astype(int)

    combined = pred.copy()
    combined[NUMERIC_FEATURES] = features[NUMERIC_FEATURES].reset_index(drop=True)
    combined["machine_type"] = one_hot.idxmax(axis=1).str.replace("Type_", "", regex=False)
    return combined, expected_count


def calculate_summaries(df, inspection_count):
    positive = df["actual_failure"].eq(1)
    n_failures = int(positive.sum())
    summary = []
    per_type = []
    for model in MODELS:
        flagged = df[f"{model}_inspection_flag"].eq(1)
        tp = int((positive & flagged).sum())
        fp = int((~positive & flagged).sum())
        fn = int((positive & ~flagged).sum())
        tn = int((~positive & ~flagged).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary.append({
            "model": model,
            "inspection_percent": BUDGET_FRACTION * 100,
            "inspections": inspection_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        for machine_type in TYPE_ORDER:
            in_type = df["machine_type"].eq(machine_type)
            type_positives = in_type & positive
            type_flagged = in_type & flagged
            type_tp = int((type_positives & flagged).sum())
            type_fp = int((in_type & ~positive & flagged).sum())
            type_fn = int((type_positives & ~flagged).sum())
            per_type.append({
                "model": model,
                "machine_type": machine_type,
                "records_in_type": int(in_type.sum()),
                "actual_failures_in_type": int(type_positives.sum()),
                "flagged_for_inspection_in_type": int(type_flagged.sum()),
                "failures_caught": type_tp,
                "missed_failures": type_fn,
                "false_alarms": type_fp,
                "recall_within_type": (
                    type_tp / int(type_positives.sum())
                    if int(type_positives.sum()) else np.nan
                ),
            })
    return pd.DataFrame(summary), pd.DataFrame(per_type)


def compare_models(df):
    positive = df["actual_failure"].eq(1)
    pairwise = []
    for i, first in enumerate(MODEL_KEYS):
        for second in MODEL_KEYS[i + 1:]:
            a = df[f"{first}_inspection_flag"].eq(1)
            b = df[f"{second}_inspection_flag"].eq(1)
            pairwise.append({
                "model_a": first,
                "model_b": second,
                "shared_flagged_observations": int((a & b).sum()),
                "flagged_by_a_only": int((a & ~b).sum()),
                "flagged_by_b_only": int((b & ~a).sum()),
                "shared_detected_failures": int((positive & a & b).sum()),
                "shared_missed_failures": int((positive & ~a & ~b).sum()),
                "failures_detected_by_a_only": int((positive & a & ~b).sum()),
                "failures_detected_by_b_only": int((positive & ~a & b).sum()),
            })
    failures = df.loc[positive].copy()
    bit_columns = [f"{model}_inspection_flag" for model in MODEL_KEYS]
    failures["detection_pattern"] = failures[bit_columns].astype(str).agg("".join, axis=1)
    failures["detected_by_n_models"] = failures[bit_columns].sum(axis=1)
    failures["missed_by_all_three"] = failures["detected_by_n_models"].eq(0)
    failures["missed_by_both_nonlinear"] = (
        failures["rbf_svm_inspection_flag"].eq(0)
        & failures["random_forest_inspection_flag"].eq(0)
    )
    failures["detection_pattern_label"] = failures["detection_pattern"].map(PATTERN_NAMES)
    detail_columns = [
        "test_row", "machine_type", *NUMERIC_FEATURES,
        *bit_columns, "detected_by_n_models", "detection_pattern",
        "detection_pattern_label", "missed_by_all_three", "missed_by_both_nonlinear",
    ]
    # These scores have different meanings/scales (SVM decision values versus
    # probabilities) and MUST NOT be directly compared across model types.
    detail_columns += [f"{model}_score" for model in MODELS]
    case_details = failures[detail_columns].copy()
    pattern_counts = pd.DataFrame([
        {"detection_pattern": key, "interpretation": label,
         "actual_failure_count": int((failures["detection_pattern"] == key).sum())}
        for key, label in PATTERN_NAMES.items()
    ])
    return pd.DataFrame(pairwise), case_details, pattern_counts


def calculate_sensor_profiles(cases):
    # RF-caught vs RF-missed groups partition all actual failures. Descriptive only.
    profiles = []
    for group, mask in (
        ("random_forest_detected", cases["random_forest_inspection_flag"].eq(1)),
        ("random_forest_missed", cases["random_forest_inspection_flag"].eq(0)),
    ):
        subset = cases.loc[mask]
        for col in NUMERIC_FEATURES:
            values = subset[col]
            profiles.append({
                "failure_group": group,
                "sensor": col,
                "failure_observations": len(subset),
                "median": values.median(),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
            })
    return pd.DataFrame(profiles)


def verify_with_existing_metrics(summary):
    if not EXISTING_METRICS_FILE.exists():
        return
    previous = pd.read_csv(EXISTING_METRICS_FILE)
    for _, s in summary.iterrows():
        match = previous.loc[(previous["model"] == s["model"])
                             & (previous["policy"] == "top_5_percent")]
        if len(match) != 1:
            raise ValueError(f"Expected existing top_5_percent row for {s['model']}")
        original = match.iloc[0]
        for our_key, previous_key in (
            ("true_positives", "failures_caught"),
            ("false_positives", "false_alarms"),
            ("false_negatives", "missed_failures"),
        ):
            if int(s[our_key]) != int(original[previous_key]):
                raise ValueError(f"Results differ from prior evaluation for {s['model']}")


def plot_detection_patterns(patterns):
    values = patterns.loc[patterns["actual_failure_count"] > 0].copy()
    values = values.sort_values("actual_failure_count", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(values["interpretation"], values["actual_failure_count"])
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", ha="left", va="center", fontsize=10)
    ax.set_xlim(0, max(values["actual_failure_count"]) * 1.14)
    ax.set_xlabel("Number of actual failure observations")
    ax.set_title("Which models detect each failure? (5% inspection budget)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "failure_detection_overlap.png", dpi=200)
    plt.close(fig)


def plot_recall_by_machine_type(per_type):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    pos = np.arange(len(TYPE_ORDER))
    bar_width = 0.25
    for i, (model, label) in enumerate(MODELS.items()):
        subset = per_type.set_index(["model", "machine_type"]).loc[
            [(model, machine_type) for machine_type in TYPE_ORDER]
        ]
        vals = subset["recall_within_type"].fillna(0).to_numpy() * 100
        ax.bar(pos + (i - 1) * bar_width, vals, width=bar_width, label=label)
    denominators = (
        per_type.loc[per_type["model"] == MODEL_KEYS[0]]
        .set_index("machine_type")["actual_failures_in_type"]
    )
    ax.set_xticks(pos)
    ax.set_xticklabels([f"Type {t} (n={int(denominators[t])} failures)"
                        for t in TYPE_ORDER])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Recall within machine type (%)")
    ax.set_title("Failure recall by machine type (5% overall budget)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.015,
             "Descriptive only; machine-type H contains very few failure cases.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(FIGURES_DIR / "failure_recall_by_machine_type.png", dpi=200)
    plt.close(fig)


def main():
    data, inspection_count = read_inputs()
    summary, per_type = calculate_summaries(data, inspection_count)
    overlaps, cases, patterns = compare_models(data)
    profiles = calculate_sensor_profiles(cases)
    verify_with_existing_metrics(summary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "failure_error_summary.csv": summary,
        "failure_error_overlap.csv": overlaps,
        "failure_detection_patterns.csv": patterns,
        "failure_error_by_type.csv": per_type,
        "failure_case_details.csv": cases,
        "failure_sensor_profiles.csv": profiles,
    }
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False)
    plot_detection_patterns(patterns)
    plot_recall_by_machine_type(per_type)

    print(f"Test set: {len(data)} observations; {int(data['actual_failure'].sum())} failures")
    print(f"Fixed 5% budget: {inspection_count} inspections per model")
    print("\nModel error summary:")
    print(summary[["model", "true_positives", "false_positives", "false_negatives", "recall"]]
          .round(4).to_string(index=False))
    print("\nPairs of models:")
    print(overlaps.to_string(index=False))
    print("\nDetection patterns (actual failures only):")
    print(patterns.loc[patterns["actual_failure_count"] > 0].to_string(index=False))
    print("\nMissed by all three models:", int(cases["missed_by_all_three"].sum()))
    print("Saved 6 CSV files to:", OUTPUT_DIR)
    print("Saved 2 figures to:", FIGURES_DIR)
    print("NOTE: Held-out test analysis is descriptive, not a basis for model retuning.")


if __name__ == "__main__":
    main()
