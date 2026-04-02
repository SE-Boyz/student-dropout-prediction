import sys
import os
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Ensure project root is in path
sys.path.append(os.path.abspath(os.getcwd()))

from utils import train_and_evaluate, ensure_directories


def train_all():
    ensure_directories()

    print("--- Training Logistic Regression ---")
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    train_and_evaluate(model=log_reg, model_name="Logistic Regression")

    print("\n--- Training Decision Tree Baseline ---")
    dt_baseline = DecisionTreeClassifier(random_state=42)
    train_and_evaluate(model=dt_baseline, model_name="Decision Tree Baseline")

    print("\n--- Training Decision Tree Optimized ---")
    dt_optimized = DecisionTreeClassifier(random_state=42)
    param_grid_dt = {
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    }
    train_and_evaluate(
        model=dt_optimized,
        model_name="Decision Tree Optimized",
        param_grid=param_grid_dt,
    )

    print("\n--- Training Random Forest Baseline ---")
    rf_baseline = RandomForestClassifier(
        random_state=42, n_jobs=1, class_weight="balanced"
    )
    train_and_evaluate(model=rf_baseline, model_name="Random Forest Baseline")

    print("\n--- Training Random Forest Optimized ---")
    rf_optimized = RandomForestClassifier(
        random_state=42, n_jobs=1, class_weight="balanced"
    )
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    train_and_evaluate(
        model=rf_optimized,
        model_name="Random Forest Optimized",
        param_grid=param_grid_rf,
    )

    print("\n--- Training KNN Baseline ---")
    knn_baseline = KNeighborsClassifier()
    train_and_evaluate(model=knn_baseline, model_name="KNN Baseline")

    print("\n--- Training KNN Optimized ---")
    knn_optimized = KNeighborsClassifier()
    param_grid_knn = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    train_and_evaluate(
        model=knn_optimized, model_name="KNN Optimized", param_grid=param_grid_knn
    )

    print("\nAll models trained successfully!")


if __name__ == "__main__":
    train_all()
