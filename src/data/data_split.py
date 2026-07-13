"""
Script for creating the data splits
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
        raw_data_path: str = "./data/raw_data/raw.csv",
        processed_data_dir: str = "./data/processed_data",
        test_size: float = 0.2,
        random_state: int = 42
        ) -> None:
    
    # Read raw data
    df = pd.read_csv(raw_data_path)

    # Split df into features X and target y
    X = df.drop(columns=["date", "silica_concentrate"])
    y = df["silica_concentrate"]

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Save train and test sets
    for file, filename in zip([X_train, X_test, y_train, y_test], ["X_train", "X_test", "y_train", "y_test"]):
        output_filepath = Path(processed_data_dir) / f"{filename}.csv"
        if not output_filepath.exists():
            file.to_csv(output_filepath, index=False)
            print(f"Split {filename} created and saved under '{output_filepath}'.")
        else:
            print(f"'{output_filepath}' already exists, skipping split creation.")


if __name__ == '__main__':
    create_splits()