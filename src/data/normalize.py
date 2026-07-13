"""
Script for feature normalization
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def normalize_data(
        processed_data_dir: str = "./data/processed_data"
        ) -> None:
    
    scaler = StandardScaler()

    filenames = ["X_train", "X_test"]

    for filename in filenames:
        # Read data
        file_path = Path(processed_data_dir) / f"{filename}.csv"
        df = pd.read_csv(file_path)
        
        # Get normalization params
        if filename == "X_train":
            scaler.fit(df)

        # Apply normalization
        df_scaled = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )

        # Save data
        output_filepath = Path(processed_data_dir) / f"{filename}_scaled.csv"

        if not output_filepath.exists():
            df_scaled.to_csv(output_filepath, index=False)
            print(f"{filename} scaled and saved under '{output_filepath}'.")
        else:
            print(f"'{output_filepath}' already exists, skipping scaling.")

   
if __name__ == "__main__":
    normalize_data()