"""
Script to evaluate a trained model
"""

import json
import joblib
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_model(
        processed_data_dir: str = "./data/processed_data", 
        model_dir: str = "./models",
        prediction_dir: str = "./data",
        metrics_dir: str = "./metrics"
        ) -> None:
    
    # Read data
    X_test_path = Path(processed_data_dir) / "X_test_scaled.csv"
    X_test = pd.read_csv(X_test_path)
    y_test_path = Path(processed_data_dir) / "y_test.csv"
    y_test = pd.read_csv(y_test_path).squeeze("columns")

    # Load model
    model_path = Path(model_dir) / "model.pkl"
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred) ** 0.5,
        "R2": r2_score(y_test, y_pred)
    }

    # Save prediction
    prediction = pd.DataFrame({
        "target": y_test,
        "prediction": y_pred
    })
    pred_path = Path(prediction_dir) / "prediction.csv"
    prediction.to_csv(pred_path, index=False)
    print(f"Prediction saved under '{pred_path}'.")

    # Save metrics
    metrics_path = Path(metrics_dir) / "scores.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Prediction metrics saved under '{metrics_path}'.")


if __name__ == '__main__':
    eval_model()