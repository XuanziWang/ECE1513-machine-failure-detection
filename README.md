# Machine Checkup

**Machine failure detection under limited maintenance inspection capacity**
ECE1513 — Introduction to Machine Learning, Fall 2026

## Project overview

This project uses supervised binary classification to prioritize machine observations for maintenance inspection. The engineering question is: **How many actual machine failures can be identified when a maintenance team can inspect only a limited fraction of observations?**

We use the [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), a **synthetic** benchmark with 10,000 records and 339 failure labels. The inputs include air and process temperature, rotational speed, torque, tool wear, and machine type. We exclude `UDI`, `Product ID`, the target `Machine failure`, and the failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) from the predictors to avoid target leakage.

The project detects failures labeled in the observations; the dataset does **not** establish how far in advance an alert could predict an impending failure.

## Models and evaluation

We compare three classifiers:

1. **Logistic regression:** linear baseline.
2. **RBF-kernel support vector machine (SVM):** nonlinear classifier.
3. **Random forest:** nonlinear ensemble classifier.

The original experiment uses an 80/20 stratified train/test split (`random_state=1513`). Models are selected with stratified five-fold cross-validation on training data, using **average precision** as the hyperparameter-selection metric. Scaling for logistic regression and SVM is fitted within each training fold. Class weights are included among the tested settings; the selected configurations in the original run use `class_weight=None`.

In addition to default-threshold precision, recall, and F1, we rank observations by each model's risk/decision score and flag the top **5%** (100 out of 2,000 test observations) for the primary comparison. We also evaluate inspection budgets of **1%, 3%, 5%, 10%, and 20%**. The 5% budget is an illustrative engineering assumption, not a measured staffing limit.

## Results

### Original held-out test split — 5% inspection budget

The 2,000 test records include 68 actual failures.

| Model | Inspections | Failures caught | False alarms | Recall | Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 100 | 39 | 61 | 57.35% | 39% |
| RBF SVM | 100 | 57 | 43 | 83.82% | 57% |
| Random forest | 100 | 57 | 43 | 83.82% | 57% |

Source: `results/evaluation/inspection_budget_metrics.csv`.

### Inspection-capacity sensitivity analysis

`src/inspection_budget_analysis.py` compares recall, precision, F1, and false alarms at the five inspection budgets, using the saved test predictions without retraining models. Results are in `results/evaluation/inspection_capacity_sensitivity.csv`; figures are in `results/figures/` (`inspection_capacity_vs_recall.png`, `inspection_capacity_vs_precision.png`, and `false_alarms_vs_capacity.png`).

Increasing the inspection budget generally catches more failures but requires more inspections and tends to lower precision. These are descriptive results for the original test split.

### Hypothetical maintenance cost analysis

`src/cost_benefit_analysis.py` uses the inspection-capacity results to study illustrative costs:

`Total cost = (inspections × cost per inspection) + (missed failures × cost per missed failure)`

The analysis assumes **CAD $100 per inspection**, with hypothetical missed-failure costs of CAD $500, $1,000, $2,000, $3,500, $5,000, and $10,000. For random forest, raising inspection capacity from 5% to 10% requires 100 additional inspections and catches three additional failures; its missed-failure **break-even cost is approximately CAD $3,333.33**. Outputs include `hypothetical_cost_analysis.csv`, `inspection_budget_break_even.csv`, and the two `hypothetical_cost_vs_capacity_missed_*.png` figures.

**These costs are assumptions, not observed factory costs or demonstrated savings.**

### Failure error analysis

`src/failure_error_analysis.py` describes missed failures, false alarms, detection overlap, and machine-type/sensor profiles for the original held-out test split at 5% inspection capacity. Among the 68 actual failures:

- 38 were detected by all three models.
- 19 were detected by both nonlinear models but missed by logistic regression.
- 1 was detected only by logistic regression.
- 10 were missed by all three models.

The SVM and random forest detected the same 57 actual failures but did **not** flag exactly the same 100 observations for inspection. Detailed CSVs are in `results/evaluation/failure_*.csv`; figures include `failure_detection_overlap.png` and `failure_recall_by_machine_type.png`.

### Multi-seed stability analysis (updated November 25, 2026)

`src/seed_stability_analysis.py` repeats the training and evaluation workflow for five stratified train/test splits (`42`, `123`, `456`, `789`, `1513`). The enhanced run completed **1,300 cross-validation fits**. At the primary 5% budget:

| Model | Mean recall across 5 seeds | Sample standard deviation |
| --- | ---: | ---: |
| Logistic regression | 54.41% | 3.12 percentage points |
| RBF SVM | 81.76% | 3.05 percentage points |
| Random forest | 80.29% | 2.46 percentage points |

Source: `results/seed_analysis/summary.csv`. The shading in `results/seed_analysis/mean_recall_vs_capacity.png` represents **±1 sample standard deviation across the five splits**, not a confidence interval. The splits overlap and are not independent field deployments.

The enhanced run also saves each seed's best parameters, cross-validation results, fold assignments, test predictions, and inspection metrics under `results/seed_analysis/seed_<seed>/`, plus dataset hash and package versions in `results/seed_analysis/run_metadata.json`.

## Repository structure

```text
 data/ai4i2020.csv                    # Synthetic input dataset
 data/processed/                     # Single-split features and labels
 src/preprocessing.py                # Split and encode data
 src/validation.py                   # CV and preprocessing pipelines
 src/tune_models.py                  # Model selection and fitting
 src/evaluate_models.py              # Single-split held-out evaluation
 src/inspection_budget_analysis.py   # Inspection-capacity sensitivity
 src/cost_benefit_analysis.py        # Hypothetical costs and break-even points
 src/failure_error_analysis.py       # Descriptive test-error analysis
 src/seed_stability_analysis.py      # Five-seed evaluation (computationally intensive)
 src/pca_analysis.py                 # Exploratory PCA; not used in final models
 results/tuning/                     # Cross-validation and fitted model artifacts
 results/evaluation/                 # Single-split metrics and analyses
 results/seed_analysis/              # Per-seed artifacts, aggregate summary, metadata
 results/figures/                    # Report-ready figures
```

## How to run (Windows PowerShell)

Run commands from the repository root. The repository includes existing data, saved model artifacts, and output CSVs/figures, so **you do not need to rerun model training just to inspect the results**.

### 1. Set up an isolated environment

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib scipy scikit-learn==1.8.0 joblib
```

If you already have a working `.venv`, use it instead of recreating it. The enhanced multi-seed run recorded Python 3.14.2, NumPy 2.4.2, pandas 3.0.2, scikit-learn 1.8.0, joblib 1.5.3, and Matplotlib 3.10.8; see `results/seed_analysis/run_metadata.json` for the exact environment and dataset hash. Loading saved `.joblib` estimators with different scikit-learn versions may be incompatible.

### 2. Regenerate the analyses from existing saved predictions (no model training)

```powershell
.\.venv\Scripts\python.exe src\inspection_budget_analysis.py
.\.venv\Scripts\python.exe src\cost_benefit_analysis.py
.\.venv\Scripts\python.exe src\failure_error_analysis.py
```

The order matters: cost analysis reads the inspection-capacity CSV. Failure error analysis requires the saved `X_test.csv`, `y_test.csv`, and `test_predictions.csv` to correspond to the same split and row order.

### 3. Full original experiment (optional; retrains models)

```powershell
.\.venv\Scripts\python.exe src\preprocessing.py
.\.venv\Scripts\python.exe src\validation.py
.\.venv\Scripts\python.exe src\tune_models.py
.\.venv\Scripts\python.exe src\evaluate_models.py
```

`src/pca_analysis.py` can be run after preprocessing as optional exploratory analysis. Model tuning performs **260 cross-validation fits**; it is not needed when using the committed artifacts.

### 4. Repeat the five-seed experiment (optional; computationally intensive)

```powershell
.\.venv\Scripts\python.exe src\seed_stability_analysis.py
```

This reruns tuning across five seeds (1,300 cross-validation fits) and writes results under `results/seed_analysis/`. Use the committed outputs unless a full reproduction is necessary.

## Limitations and interpretation

- AI4I is **synthetic**, so results are benchmark findings rather than validated performance in a real maintenance program.
- Only **3.39%** of records carry a failure label; overall accuracy alone would be misleading.
- The 5% inspection budget and all CAD costs are **illustrative assumptions**.
- Original held-out test error analysis is **descriptive** and must not be used to retune models while claiming the same test set remains an untouched evaluation.
- Multi-seed standard deviations describe differences across overlapping random splits; they are not deployment uncertainty estimates or confidence intervals.
- Additional work could compare a training-defined rule-based baseline or evaluate the approach on independently collected operational data.

## Data source

UCI Machine Learning Repository: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).

## Development history

The entries below retain the original team progress notes (apart from correcting `Nov 23th` to `Nov 23rd`). The additional dated analysis notes are placed afterward to keep authors' work identifiable.

## By Nov 25th

- Enhanced the seed analysis to save each seed's best parameters, cross-validation results, predictions, and fold assignments.
- Recorded the dataset hash and package versions, and labeled the graph's standard-deviation shading.
- Reran all five seeds (1,300 cross-validation fits) and verified the exported results. Mean recall at the assumed 5% inspection capacity remained 81.76% for SVM, 80.29% for Random Forest, and 54.41% for Logistic Regression.

## Nov 23rd
Added model evaluation based on multiple seeds to calculate mean recall and std- inspired by Part 3a in assignment 4. 
Conclusion: Across five random seeds, RBF SVM achieved the highest average recall (81.76%) at the 5% inspection capacity, followed closely by Random Forest (80.29%), while Logistic Regression achieved 54.41%. The inspection capacity analysis also showed a similar trend to the original single-seed results: increasing inspection capacity allowed more failures to be detected, but also led to more false alarms. Overall, the multi-seed analysis supports the conclusions from the original evaluation.

## By Nov 19th

- Set up stratified five-fold cross-validation with scaling fitted inside each training fold.
- Tuned logistic regression, RBF SVM, and random forest across 260 cross-validation fits, selecting settings by average precision.
- Evaluated the tuned models on 2,000 held-out test observations containing 68 failures.
- Assumed a 5% inspection budget (100 observations). SVM and random forest each caught 57 failures (83.8% recall), compared with 39 for logistic regression (57.4% recall).
- Saved and verified the training/evaluation scripts, fitted models, cross-validation results, and test predictions.
- Some possible extensions: a rule-based baseline, an inspection-budget graph, and a hypothetical cost analysis.

## Additional analyses — Nov 19 and Nov 21
These analyses extend the original 5% inspection-budget evaluation using the saved held-out test predictions. They do not retrain or retune the three classifiers.

### Nov 19 — Inspection capacity sensitivity analysis

- Compared logistic regression, RBF SVM, and random forest at inspection budgets of 1%, 3%, 5%, 10%, and 20%.
- Evaluated failures detected, missed failures, false alarms, precision, recall, and F1-score.
- Saved `src/inspection_budget_analysis.py`, `results/evaluation/inspection_capacity_sensitivity.csv`, and comparison figures in `results/figures/`.
- [PR #1](https://github.com/XuanziWang/ECE1513-machine-failure-detection/pull/1)

### Nov 19 — Hypothetical maintenance cost analysis

- Examined trade-offs between inspection workload and missed failures using illustrative costs, not observed maintenance expenses.
- Used a hypothetical CAD $100 per inspection and varied the assumed cost per missed failure.
- Calculated break-even costs for increasing inspection capacity from 5% to 10%; for random forest, the illustrative threshold was approximately CAD $3,333.33 per missed failure.
- Saved `src/cost_benefit_analysis.py`, cost results, break-even results, and figures.
- [PR #2](https://github.com/XuanziWang/ECE1513-machine-failure-detection/pull/2)

### Nov 21 — Failure error analysis

- Analyzed false positives, false negatives, missed failures by machine type, and detection overlap under the fixed 5% inspection budget.
- Of 68 actual failures, all three models detected 38, SVM and random forest alone detected 19, logistic regression alone detected one, and all three missed 10.
- Saved `src/failure_error_analysis.py`, six CSV tables, and two figures.
- [PR #3](https://github.com/XuanziWang/ECE1513-machine-failure-detection/pull/3)

**Note:** The inspection-budget graphs and hypothetical cost analysis listed earlier as possible extensions have now been completed. All results are based on a synthetic benchmark; the cost analysis is illustrative, and test-set error analysis is descriptive only.
