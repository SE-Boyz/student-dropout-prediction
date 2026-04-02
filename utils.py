"""
Student Dropout Prediction Utilities

This module provides utility functions for data loading, preprocessing,
model training, and evaluation for the Student Dropout Prediction project.
It defines standard paths, categorical columns, and machine learning pipelines
to ensure consistency across notebooks and scripts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Check if we are running in an interactive environment (like a Jupyter notebook)
# If not, use the 'Agg' backend for script-based figure saving.
try:
    if "ipykernel" not in sys.modules:
        matplotlib.use("Agg")
except (AttributeError, ImportError):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_FILENAME = "student_dropout_academic_success.csv"
DATASET_URL = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"
DATASET_DOI = "https://doi.org/10.24432/C5MC89"
TARGET_COLUMN = "target"
TARGET_LABEL_ORDER = ["Dropout", "Enrolled", "Graduate"]

# -----------------------------------------------------------------------------
# Feature Categorization
# -----------------------------------------------------------------------------
# Integer-coded categories should be treated as nominal values rather than scaled.
CATEGORICAL_COLUMNS = [
    "marital_status",
    "application_mode",
    "course",
    "daytime_evening_attendance",
    "previous_qualification",
    "nacionality",
    "mother's_qualification",
    "father's_qualification",
    "mother's_occupation",
    "father's_occupation",
    "displaced",
    "educational_special_needs",
    "debtor",
    "tuition_fees_up_to_date",
    "gender",
    "scholarship_holder",
    "international",
]


def ensure_directories() -> None:
    """Ensures that all necessary project directories exist."""
    for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def dataset_path() -> Path:
    return DATA_DIR / DATASET_FILENAME


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names by stripping whitespace, converting to lowercase,
    and replacing spaces/hyphens with underscores.
    """
    df = df.copy()
    df.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in df.columns
    ]
    return df


def load_student_dataset() -> tuple[pd.DataFrame, Path]:
    """
    Loads the student dropout dataset from the data directory.

    Returns:
        tuple: (DataFrame containing the dataset, Path to the dataset file)

    Raises:
        FileNotFoundError: If the dataset file is missing.
        ValueError: If the target column is missing from the dataset.
    """
    ensure_directories()
    path = dataset_path()
    if not path.exists():
        raise FileNotFoundError(
            "The dataset file was not found. Download it from the UCI repository and "
            f"place it at {path}."
        )

    # Read CSV and handle common null value representations
    df = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])
    df = normalize_columns(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected '{TARGET_COLUMN}' column in {path}.")

    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    return df, path


def split_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separates categorical and numerical feature names."""
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    categorical_columns = [
        column for column in feature_columns if column in CATEGORICAL_COLUMNS
    ]
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    return categorical_columns, numeric_columns


def inspect_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Returns a brief analysis of the dataset including class distribution,
    missing values, and numeric summaries.
    """
    categorical_columns, numeric_columns = split_feature_types(df)

    # Analyze target class balance
    class_distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .reindex(TARGET_LABEL_ORDER)
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = (
        class_distribution["count"] / class_distribution["count"].sum()
    ).round(4)

    numeric_summary = df[numeric_columns].describe().transpose().round(2)

    # Create an overview of features and their assigned roles/types
    feature_overview = pd.DataFrame(
        {
            "feature_name": df.columns,
            "role": [
                "target" if column == TARGET_COLUMN else "feature"
                for column in df.columns
            ],
            "feature_type": [
                (
                    "categorical"
                    if column in categorical_columns
                    else "numeric" if column != TARGET_COLUMN else "target"
                )
                for column in df.columns
            ],
        }
    )

    return {
        "data_types": pd.DataFrame(df.dtypes, columns=["dtype"]),
        "missing_values": pd.DataFrame(df.isna().sum(), columns=["missing_count"]),
        "class_distribution": class_distribution,
        "numeric_summary": numeric_summary,
        "feature_overview": feature_overview,
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer for preprocessing features.

    Numeric features: Imputed with median and scaled using StandardScaler.
    Categorical features: Imputed with most frequent and encoded with OneHotEncoder.
    """
    categorical_columns = [
        column for column in X.columns if column in CATEGORICAL_COLUMNS
    ]
    numeric_columns = [
        column for column in X.columns if column not in categorical_columns
    ]

    # Handle Scikit-learn version differences for sparse output
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def preprocess_and_save(
    test_size: float = 0.2, random_state: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Loads dataset, performs train-test split, fits preprocessor,
    and saves processed CSVs and metadata.
    """
    ensure_directories()
    df, source_path = load_student_dataset()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Use stratified split to maintain class balance in training and testing sets
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    # Convert arrays back to DataFrames with proper feature names
    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train = pd.DataFrame(
        X_train_array, columns=feature_names, index=X_train_raw.index
    )
    X_test = pd.DataFrame(X_test_array, columns=feature_names, index=X_test_raw.index)
    y_train_df = pd.DataFrame({TARGET_COLUMN: y_train})
    y_test_df = pd.DataFrame({TARGET_COLUMN: y_test})

    # Save artifacts
    X_train.to_csv(DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(DATA_DIR / "X_test.csv", index=False)
    y_train_df.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test_df.to_csv(DATA_DIR / "y_test.csv", index=False)

    metadata = {
        "dataset_path": str(source_path),
        "dataset_url": DATASET_URL,
        "doi": DATASET_DOI,
        "instances": len(df),
        "original_features": X.shape[1],
        "encoded_features": len(feature_names),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "categorical_feature_count": len(
            [column for column in X.columns if column in CATEGORICAL_COLUMNS]
        ),
        "numeric_feature_count": len(
            [column for column in X.columns if column not in CATEGORICAL_COLUMNS]
        ),
        "class_distribution": y.value_counts().reindex(TARGET_LABEL_ORDER).to_dict(),
        "random_state": random_state,
        "test_size": test_size,
    }

    (RESULTS_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    joblib.dump(preprocessor, RESULTS_DIR / "preprocessor.joblib")

    return {
        "raw_dataset": df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train_df,
        "y_test": y_test_df,
    }


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads pre-split and processed CSV files.
    Triggers preprocessing if files are missing.
    """
    required_files = [
        DATA_DIR / "X_train.csv",
        DATA_DIR / "X_test.csv",
        DATA_DIR / "y_train.csv",
        DATA_DIR / "y_test.csv",
    ]
    if not all(path.exists() for path in required_files):
        preprocess_and_save()

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    """Calculates comprehensive classification metrics for a set of predictions."""
    report = classification_report(
        y_true,
        y_pred,
        labels=TARGET_LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_LABEL_ORDER)

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision (weighted)": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall (weighted)": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1-score (weighted)": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1-score (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Confusion Matrix": cm.tolist(),
        "Labels": TARGET_LABEL_ORDER,
        "Classification Report": report,
    }


def save_confusion_matrix(
    model_name: str, y_true: pd.Series, y_pred: np.ndarray
) -> Path:
    """Generates and saves a confusion matrix heatmap as an image."""
    cm = confusion_matrix(y_true, y_pred, labels=TARGET_LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=TARGET_LABEL_ORDER
    )
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()

    figure_path = (
        FIGURES_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def run_cross_validation(model) -> dict[str, float]:
    """Runs 5-fold stratified cross-validation on the training set."""
    X_train, _, y_train, _ = load_processed_data()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_weighted": "f1_weighted",
        },
        n_jobs=1,
    )
    return {
        "CV Accuracy Mean": float(scores["test_accuracy"].mean()),
        "CV Accuracy Std": float(scores["test_accuracy"].std()),
        "CV Balanced Accuracy Mean": float(scores["test_balanced_accuracy"].mean()),
        "CV Weighted F1 Mean": float(scores["test_f1_weighted"].mean()),
    }


def train_and_evaluate(
    model, model_name: str, param_grid: dict | None = None
) -> tuple[dict, pd.DataFrame]:
    """
    Trains a model (with optional hyperparameter tuning), evaluates it on the test set,
    and persists both the model and its metrics.

    Args:
        model: The estimator object.
        model_name: Display name for the algorithm.
        param_grid: Optional dictionary for GridSearchCV.

    Returns:
        tuple: (Combined metrics dictionary, Single-row summary DataFrame)
    """
    ensure_directories()
    X_train, X_test, y_train, y_test = load_processed_data()

    selected_model = model
    tuning_summary: dict[str, object] = {}

    # Perform hyperparameter optimization if a grid is provided
    if param_grid:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        selected_model = search.best_estimator_
        tuning_summary = {
            "Best Parameters": search.best_params_,
            "Best CV Weighted F1": float(search.best_score_),
        }
    else:
        selected_model.fit(X_train, y_train)

    predictions = selected_model.predict(X_test)

    # Calculate training accuracy for overfitting assessment
    train_accuracy = float(selected_model.score(X_train, y_train))

    # Collect and save result artifacts
    metrics = evaluate_predictions(y_test, predictions)
    metrics.update(tuning_summary)
    metrics.update(run_cross_validation(selected_model))
    metrics["Training Accuracy"] = train_accuracy
    metrics["Model"] = model_name
    metrics["Confusion Matrix Figure"] = str(
        save_confusion_matrix(model_name, y_test, predictions)
    )

    metrics_path = METRICS_DIR / f"{model_name.lower().replace(' ', '_')}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(
        selected_model, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    metrics_df = pd.DataFrame(
        [
            {
                "Algorithm": model_name,
                "Training Accuracy": train_accuracy,
                "Accuracy": metrics["Accuracy"],
                "Balanced Accuracy": metrics["Balanced Accuracy"],
                "Weighted F1": metrics["F1-score (weighted)"],
                "Macro F1": metrics["F1-score (macro)"],
                "CV Accuracy Mean": metrics["CV Accuracy Mean"],
            }
        ]
    )
    return metrics, metrics_df


def create_comparison_table() -> pd.DataFrame:
    """Aggregates all saved model metrics into a single comparison DataFrame."""
    ensure_directories()
    metric_files = sorted(METRICS_DIR.glob("*.json"))
    if not metric_files:
        raise FileNotFoundError(
            "No saved metrics were found. Run the individual model notebooks first."
        )

    rows = []
    for file_path in metric_files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "Algorithm": payload["Model"],
                "Training Accuracy": payload.get("Training Accuracy", "N/A"),
                "Accuracy": payload["Accuracy"],
                "Balanced Accuracy": payload["Balanced Accuracy"],
                "Weighted Precision": payload["Precision (weighted)"],
                "Weighted Recall": payload["Recall (weighted)"],
                "Weighted F1": payload["F1-score (weighted)"],
                "Macro F1": payload["F1-score (macro)"],
                "CV Accuracy Mean": payload["CV Accuracy Mean"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["Weighted F1", "Balanced Accuracy"], ascending=False)
        .reset_index(drop=True)
    )


def plot_model_comparison(comparison_df: pd.DataFrame):
    """Generates a bar chart comparing performance metrics across different algorithms."""
    plot_df = comparison_df.melt(
        id_vars="Algorithm",
        value_vars=["Accuracy", "Balanced Accuracy", "Weighted F1", "Macro F1"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Algorithm", y="Score", hue="Metric", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xlabel("Algorithm")
    ax.legend(loc="lower right")
    fig.tight_layout()

    figure_path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(figure_path, dpi=150)
    return ax
