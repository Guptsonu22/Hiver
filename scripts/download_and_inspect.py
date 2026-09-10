import os
import sys
import pandas as pd
from huggingface_hub import hf_hub_download

os.makedirs("data/raw", exist_ok=True)
print("Downloading parquet dataset from HuggingFace...")
path = hf_hub_download(
    repo_id="gorkemsevinc/Customer_Support_on_Twitter",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir="data/raw"
)
print("Downloaded to:", path)
df = pd.read_parquet(path)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
