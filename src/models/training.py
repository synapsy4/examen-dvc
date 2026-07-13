"""
Script for training
"""

import joblib
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_model(       
        processed_data_dir: str = "./data/processed_data", 
        model_dir: str = "./models",
        random_state: int = 42
        ) -> None:
    
    # Read data
    X_train_path = Path(processed_data_dir) / "X_train_scaled.csv"
    X_train = pd.read_csv(X_train_path)
    y_train_path = Path(processed_data_dir) / "y_train.csv"
    y_train = pd.read_csv(y_train_path).squeeze("columns")

    # Load parameters
    params_path = Path(model_dir) / "best_params.pkl"
    model_params = joblib.load(params_path)

    # Define model
    model = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **model_params
    )

    # Run training
    model.fit(X_train, y_train)

    # Save model
    model_path = Path(model_dir) / "model.pkl"
    joblib.dump(model, model_path)
    print(f"Trained model saved under '{model_path}'.")


if __name__ == '__main__':
    train_model()