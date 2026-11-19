# Machine Checkup

Machine failure detection using sensor data to prioritize maintenance inspections.

## By Nov 19th

- Set up stratified five-fold cross-validation with scaling fitted inside each training fold.
- Tuned logistic regression, RBF SVM, and random forest across 260 cross-validation fits, selecting settings by average precision.
- Evaluated the tuned models on 2,000 held-out test observations containing 68 failures.
- Assumed a 5% inspection budget (100 observations). SVM and random forest each caught 57 failures (83.8% recall), compared with 39 for logistic regression (57.4% recall).
- Saved and verified the training/evaluation scripts, fitted models, cross-validation results, and test predictions.
- Some possible extensions: a rule-based baseline, an inspection-budget graph, and a hypothetical cost analysis. 
