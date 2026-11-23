# Machine Checkup

Machine failure detection using sensor data to prioritize maintenance inspections.

## Nov 23th
Added model evaluation based on multiple seeds to calculate mean recall and std- inspired by Part 3a in assignment 4. 
Conclusion: Across five random seeds, RBF SVM achieved the highest average recall (81.76%) at the 5% inspection capacity, followed closely by Random Forest (80.29%), while Logistic Regression achieved 54.41%. The inspection capacity analysis also showed a similar trend to the original single-seed results: increasing inspection capacity allowed more failures to be detected, but also led to more false alarms. Overall, the multi-seed analysis supports the conclusions from the original evaluation.

## By Nov 19th

- Set up stratified five-fold cross-validation with scaling fitted inside each training fold.
- Tuned logistic regression, RBF SVM, and random forest across 260 cross-validation fits, selecting settings by average precision.
- Evaluated the tuned models on 2,000 held-out test observations containing 68 failures.
- Assumed a 5% inspection budget (100 observations). SVM and random forest each caught 57 failures (83.8% recall), compared with 39 for logistic regression (57.4% recall).
- Saved and verified the training/evaluation scripts, fitted models, cross-validation results, and test predictions.
- Some possible extensions: a rule-based baseline, an inspection-budget graph, and a hypothetical cost analysis. 
