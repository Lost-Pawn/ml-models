import os
import tempfile
import pandas as pd

from kaggle.api.kaggle_api_extended import KaggleApi

try:
    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory() as tmp_dir:
        print("Fetching dataset...")
        api.dataset_download_files("blastchar/telco-customer-churn", path=tmp_dir, unzip=False)
        downloaded_files = os.listdir(tmp_dir)
        target_file = downloaded_files[0]       
        file_path = os.path.join(tmp_dir, target_file)
        dataset = pd.read_csv(file_path, compression='infer')
    df = dataset.copy()  # Keep a fresh copy for later use

except Exception as e:
    print(f"An error occurred while fetching the dataset: {e}")
    
print(dataset.head())

# comment out the above part after loading the dataset once and use the copy for eda and model building