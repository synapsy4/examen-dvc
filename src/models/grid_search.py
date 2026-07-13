"""
Script for performing grid search
"""

import joblib
from pathlib import Path

import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def perform_gridsearch(
        processed_data_dir: str = "./data/processed_data", 
        model_dir: str = "./models/models",
        n_trials: int = 50,
        random_state: int = 42
        ) -> None:
    
    # Read data
    X_train_path = Path(processed_data_dir) / "X_train_scaled.csv"
    X_train = pd.read_csv(X_train_path)
    y_train_path = Path(processed_data_dir) / "y_train.csv"
    y_train = pd.read_csv(y_train_path).squeeze("columns")

    # Define model
    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1
    )

    # Define parameter search space
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 4),
    }

    # Define grid search 
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_trials,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42
    )
    
    # Run grid search
    search.fit(X_train, y_train)

    # Save best parameters
    best_params = search.best_params_
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    params_path = model_dir / "best_params.pkl"
    joblib.dump(best_params, params_path)
    print(f"Best parameters saved under '{params_path}'.")


if __name__ == '__main__':
    perform_gridsearch()