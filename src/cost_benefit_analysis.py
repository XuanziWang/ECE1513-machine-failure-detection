r"""Hypothetical maintenance cost analysis for the ECE1513 project.

Run from the repository root:
    .\.venv\Scripts\python.exe src\cost_benefit_analysis.py

Uses the already-computed inspection-capacity sensitivity CSV. Does not
retrain or select models using the held-out test set. Monetary assumptions
are illustrative only, not observed operating costs or actual savings.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a graphical display
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results" / "evaluation" / "inspection_capacity_sensitivity.csv"
OUTPUT_DIR = ROOT / "results" / "evaluation"
FIGURES_DIR = ROOT / "results" / "figures"

# HYPOTHETICAL Canadian-dollar costs; not supplied by the UCI dataset.
COST_PER_INSPECTION_CAD = 100.0
MISSED_FAILURE_COSTS_CAD = [500, 1000, 2000, 3500, 5000, 10000]
PLOT_MISSED_FAILURE_COSTS_CAD = [2000, 5000]
PRIMARY_BUDGET_PERCENT = 5.0
COMPARISON_BUDGET_PERCENT = 10.0
MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "rbf_svm": "RBF SVM",
    "random_forest": "Random Forest",
}


def read_analysis():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}. Run src/inspection_budget_analysis.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    required = {
        "model", "inspection_percent", "inspection_count", "failures_caught",
        "false_alarms", "missed_failures",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in required - {"model"}:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df.duplicated(subset=["model", "inspection_percent"]).any():
        raise ValueError("Duplicate model + budget rows in input CSV.")
    if (df["inspection_count"] != df["failures_caught"] + df["false_alarms"]).any():
        raise ValueError("Inspection counts do not match TP + FP.")
    if (df[["inspection_count", "failures_caught", "false_alarms", "missed_failures"]] < 0).any().any():
        raise ValueError("Counts must be nonnegative.")
    total_failures = df["failures_caught"] + df["missed_failures"]
    if total_failures.nunique() != 1:
        raise ValueError("Rows do not describe the same number of actual failures.")
    unknown_models = set(df["model"]) - set(MODEL_NAMES)
    if unknown_models:
        raise ValueError(f"Unexpected model names: {sorted(unknown_models)}")
    return df


def build_cost_results(df):
    scenarios = []
    for missed_cost in MISSED_FAILURE_COSTS_CAD:
        scenario = df[["model", "inspection_percent", "inspection_count",
                       "failures_caught", "false_alarms", "missed_failures"]].copy()
        scenario["cost_per_inspection_cad"] = COST_PER_INSPECTION_CAD
        scenario["cost_per_missed_failure_cad"] = missed_cost
        scenario["inspection_cost_cad"] = (
            scenario["inspection_count"] * COST_PER_INSPECTION_CAD
        )
        scenario["missed_failure_cost_cad"] = (
            scenario["missed_failures"] * missed_cost
        )
        # Do not add an extra false-alarm fee here: every flagged record
        # (true or false alarm) already incurs the inspection cost above.
        scenario["total_hypothetical_cost_cad"] = (
            scenario["inspection_cost_cad"] + scenario["missed_failure_cost_cad"]
        )
        scenarios.append(scenario)
    return pd.concat(scenarios, ignore_index=True).sort_values(
        ["cost_per_missed_failure_cad", "model", "inspection_percent"]
    )


def compare_five_to_ten(df):
    rows = []
    for model in MODEL_NAMES:
        subset = df.loc[df["model"] == model]
        five = subset.loc[subset["inspection_percent"].round(8) == PRIMARY_BUDGET_PERCENT]
        ten = subset.loc[subset["inspection_percent"].round(8) == COMPARISON_BUDGET_PERCENT]
        if len(five) != 1 or len(ten) != 1:
            raise ValueError(f"Expected 5% and 10% budget rows for {model}.")
        five, ten = five.iloc[0], ten.iloc[0]
        added_inspections = int(ten["inspection_count"] - five["inspection_count"])
        additional_failures = int(ten["failures_caught"] - five["failures_caught"])
        if added_inspections <= 0 or additional_failures < 0:
            raise ValueError(f"Unexpected inspection/failure counts for {model}.")
        threshold = (COST_PER_INSPECTION_CAD * added_inspections / additional_failures
                     if additional_failures > 0 else float("nan"))
        rows.append({
            "model": model,
            "from_budget_percent": PRIMARY_BUDGET_PERCENT,
            "to_budget_percent": COMPARISON_BUDGET_PERCENT,
            "added_inspections": added_inspections,
            "additional_failures_caught": additional_failures,
            "break_even_missed_failure_cost_cad": threshold,
        })
    return pd.DataFrame(rows)


def plot_scenario(costs, missed_cost):
    scenario = costs.loc[costs["cost_per_missed_failure_cad"] == missed_cost]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, label in MODEL_NAMES.items():
        subset = scenario.loc[scenario["model"] == model].sort_values("inspection_percent")
        ax.plot(subset["inspection_percent"], subset["total_hypothetical_cost_cad"],
                marker="o", linewidth=2, label=label)
    ax.axvline(PRIMARY_BUDGET_PERCENT, color="gray", linestyle="--", alpha=0.6,
               label="Primary 5% budget")
    ax.set_title("Hypothetical inspection + missed-failure cost")
    ax.set_xlabel("Inspection capacity (% of held-out test observations)")
    ax.set_ylabel("Total hypothetical cost (CAD)")
    ax.set_xticks(sorted(scenario["inspection_percent"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.text(0.5, 0.005,
             f"Assumptions: CAD {COST_PER_INSPECTION_CAD:,.0f} per inspection; "
             f"CAD {missed_cost:,} per missed failure; synthetic dataset",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    filename = f"hypothetical_cost_vs_capacity_missed_{missed_cost}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=200)
    plt.close(fig)
    return filename


def main():
    source = read_analysis()
    costs = build_cost_results(source)
    break_even = compare_five_to_ten(source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cost_file = OUTPUT_DIR / "hypothetical_cost_analysis.csv"
    break_even_file = OUTPUT_DIR / "inspection_budget_break_even.csv"
    costs.to_csv(cost_file, index=False)
    break_even.to_csv(break_even_file, index=False)
    filenames = [plot_scenario(costs, c) for c in PLOT_MISSED_FAILURE_COSTS_CAD]

    print("Hypothetical cost analysis (not measured maintenance costs)")
    print(f"Inspection cost: CAD ${COST_PER_INSPECTION_CAD:,.0f} per flagged observation")
    for missed_cost in PLOT_MISSED_FAILURE_COSTS_CAD:
        subset = costs.loc[
            (costs["cost_per_missed_failure_cad"] == missed_cost)
            & (costs["inspection_percent"].isin([5.0, 10.0])),
            ["model", "inspection_percent", "inspection_count", "missed_failures",
             "total_hypothetical_cost_cad"],
        ]
        print(f"\nCost per missed failure = CAD ${missed_cost:,.0f}")
        print(subset.to_string(index=False))
    print("\nBreak-even cost per missed failure for increasing 5% -> 10%:")
    print(break_even.to_string(index=False, float_format=lambda n: f"{n:,.2f}"))
    print(f"\nSaved CSV files:\n  {cost_file}\n  {break_even_file}")
    print("Saved figures:")
    for filename in filenames:
        print(f"  {FIGURES_DIR / filename}")


if __name__ == "__main__":
    main()
