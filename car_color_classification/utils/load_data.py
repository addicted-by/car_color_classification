import os
import subprocess
from pathlib import Path

from dvc.repo import Repo

from car_color_classification.logger import setup_custom_logger


# import gdown


logger = setup_custom_logger(__name__)


# TRAIN_URL = "https://drive.google.com/uc?id=17DWaO2v_bWZ6cP_4e2IqVVmc7nGx2DCL"
# VAL_URL = "https://drive.google.com/uc?id=1z5R22UfZqIrGy0UnhPI3ZErOQO9wwhpf"
# datasets = {
#     # "private_dataset" : "",
#     # "public_test" : "",
#     "train": TRAIN_URL,
#     "val": VAL_URL,
# }


def load_data():
    datasets = {"train": "train.tar.gz", "val": "val.tar.gz"}

    DATA_PATH = Path("data")
    repo = Repo(".")
    repo.pull()
    logger.info("Untar files")
    for name, dataset in datasets.items():
        if not (DATA_PATH / name).exists():
            cmd = f"tar xzf {DATA_PATH / dataset} -C {DATA_PATH}"  # | rm -f {DATA_PATH / dataset}"
            subprocess.run(cmd, shell=True)
    # DATA_PATH = Path("data")
    # if not os.path.exists(DATA_PATH) or not len(os.listdir(DATA_PATH)):
    #     for dataset, url in datasets.items():
    #         dataset_path = os.path.join(DATA_PATH, f"{dataset}.tar.gz")
    #         logger.info("Creating dir data")
    #         os.makedirs(DATA_PATH, exist_ok=True)
    #         logger.info("Loading data...")
    #         gdown.download(url, output=dataset_path)
    #         logger.info("Untar files...")
    #         cmd = f"tar xzf {dataset_path} -C {DATA_PATH}"
    #         subprocess.run(cmd, shell=True)

    logger.info("Data had already been collected.")
    logger.info(f"Check --> {DATA_PATH}")

    return (os.path.join(DATA_PATH, key.split(".")[0]) for key in datasets)
