"""Training-only cross-validation and preprocessing for model selection."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
NUMERICAL_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_training_data():
    """Load unscaled features and the matching one-dimensional target."""
    X = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Machine failure"]
    if len(X) != len(y):
        raise ValueError("Training features and labels must have matching lengths.")
    return X, y


def make_cv():
    """Reuse identical, reproducible folds across model comparisons."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=1513)


def make_model_pipeline(model, *, scale=True):
    """Fit scaling inside each CV fold; pass scale=False for random forests.

    Pass this whole pipeline to GridSearchCV with cv=make_cv(). Model
    parameter names use the 'model__' prefix, for example 'model__C'.
    """
    preprocessing = "passthrough"
    if scale:
        preprocessing = ColumnTransformer(
            [("numeric", StandardScaler(), NUMERICAL_COLUMNS)],
            remainder="passthrough",
        )
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def main():
    X, y = load_training_data()
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(
        make_cv().split(X, y), start=1
    ):
        train_labels = y.iloc[train_indices]
        validation_labels = y.iloc[validation_indices]
        rows.append({
            "fold": fold,
            "train_rows": len(train_indices),
            "train_failures": int(train_labels.sum()),
            "validation_rows": len(validation_indices),
            "validation_failures": int(validation_labels.sum()),
            "validation_failure_rate": validation_labels.mean(),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
