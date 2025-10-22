# scripts/download_proteingym.py
"""
Download and extract the ProteinGym DMS Substitution benchmark (v1.3),
then parse one assay into a tidy CSV for downstream fine-tuning.

Requires:
    pip install pandas tqdm requests
"""

import os, re, zipfile, requests
import pandas as pd
from tqdm import tqdm

OUT_DIR = "data/raw/proteingym"
os.makedirs(OUT_DIR, exist_ok=True)

# Download and unzip the official DMS Substitutions benchmark
version = "v1.3"
zip_name = "DMS_ProteinGym_substitutions.zip"
url = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{version}/{zip_name}"
zip_path = os.path.join(OUT_DIR, zip_name)

if not os.path.exists(os.path.join(OUT_DIR, "DMS_substitutions")):
    print(f"▶ Downloading {zip_name} (~1 GB)...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("▶ Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(OUT_DIR)
    os.remove(zip_path)
else:
    print("▶ DMS_substitutions already present; skipping download.")

# The unzipped folder is data/raw/proteingym/DMS_substitutions/
root = os.path.join(OUT_DIR, "DMS_substitutions")