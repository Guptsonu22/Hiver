import os
from huggingface_hub import hf_hub_download
import pandas as pd

os.makedirs("data/raw", exist_ok=True)
print("Downloading twcs.csv from SunidhiSriram/twcs...")
csv_path = hf_hub_download(
    repo_id="SunidhiSriram/twcs",
    filename="twcs.csv",
    repo_type="dataset",
    local_dir="data/raw"
)
print("Downloaded twcs.csv to:", csv_path)

# Read first 10 rows to inspect schema
df_sample = pd.read_csv(csv_path, nrows=10)
print("Columns:", df_sample.columns.tolist())
print("\nFirst 3 rows:")
print(df_sample.head(3))
