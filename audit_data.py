import os
import requests
import pandas as pd

# Official NetFlow V2 source for NF-UNSW-NB15
URL = "https://rdm.uq.edu.au/files/8c6e2a00-ef9c-11ed-827d-e762de186848"
FILENAME = "data/raw/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv"

def download_dataset():
    if os.path.exists(FILENAME):
        print(f"{FILENAME} already exists. Skipping download.")
        return

    print("Connecting to University Archive server...")
    response = requests.get(URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    print(f" File size: {total_size / (1024*1024):.2f} MB. Starting chunked stream...")

    with open(FILENAME, 'wb') as file:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                print(f" Progress: {downloaded / (1024*1024):.1f} MB downloaded", end='\r')
    print("\n Download complete!")

def audit_features():
    print("\n Loading data matrix into Pandas...")
    # Read just the first 5 rows to quickly verify columns and types
    df = pd.read_csv(FILENAME, nrows=5)

    print("\n--- DATASET PROFILE ---")
    print(f"Total Columns found: {len(df.columns)}")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    print("\nFirst Sample Row:")
    print(df.head(1).to_dict(orient='records')[0])

if __name__ == "__main__":
    audit_features()