"""
Script for downloading the raw data
"""

from pathlib import Path

import requests


def download_data(
        data_url: str = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv",
        raw_data_path: str = "./data/raw_data"
        ) -> None:
    
    file_name = Path(data_url).name
    output_file = Path(raw_data_path) / file_name

    if output_file.exists():
        print(f"File '{output_file}' already exists, skipping download.")
        return

    response = requests.get(data_url)
    if response.status_code == 200:
        # Process the response content as needed
        content = response.text
        text_file = open(output_file, "wb")
        text_file.write(content.encode('utf-8'))
        text_file.close()
        print(f"File '{output_file}' successfully downloaded.")
    else:
        print(f"Error accessing the object {file_name}:", response.status_code)


if __name__ == "__main__":
    download_data()