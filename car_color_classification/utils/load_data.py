import os
import subprocess
from pathlib import Path

import gdown


TRAIN_URL = "https://drive.google.com/uc?id=17DWaO2v_bWZ6cP_4e2IqVVmc7nGx2DCL"
VAL_URL = "https://drive.google.com/uc?id=1z5R22UfZqIrGy0UnhPI3ZErOQO9wwhpf"
datasets = {
    # "private_dataset" : "",
    # "public_test" : "",
    "train": TRAIN_URL,
    "val": VAL_URL,
}


def load_data():
    DATA_PATH = Path("data")
    if not os.path.exists(DATA_PATH) or not len(os.listdir(DATA_PATH)):
        for dataset, url in datasets.items():
            dataset_path = os.path.join(DATA_PATH, f"{dataset}.tar.gz")
            print("Creating dir data")
            os.makedirs(DATA_PATH, exist_ok=True)
            print("Loading data...")
            gdown.download(url, output=dataset_path)
            print("Untar files...")
            cmd = f"tar xzf {dataset_path} -C {DATA_PATH}"
            subprocess.run(cmd, shell=True)

    print(f"Check {DATA_PATH}")

    return (os.path.join(DATA_PATH, key) for key in datasets)
