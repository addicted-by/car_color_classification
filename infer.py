import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from natsort import natsort_keygen
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from car_color_classification.logger import setup_custom_logger
from car_color_classification.models import get_model_by_name
from car_color_classification.utils.args import parse_arguments
from car_color_classification.utils.config import load_config
from car_color_classification.utils.datasets import CarsDataset
from car_color_classification.utils.load_data import load_data


logger = setup_custom_logger(__name__)

sys.path.append("./car_color_classification")


def predict(model, test_loader, device):
    with torch.no_grad():
        logits = []
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    probs = torch.cat(logits).numpy()
    return probs


if __name__ == "__main__":
    args = parse_arguments(default="./configs/inference.yaml")
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    _, TEST_DIR = load_data()
    with open("label_encoder.pkl", "rb") as le_encoder_file:
        label_encoder = pickle.load(le_encoder_file)

    test_files = sorted(list(Path(TEST_DIR).rglob("*.jpg")))

    n_classes = len(label_encoder.classes_)

    config = load_config(
        args.config, default_arguments="./configs/base/default_arguments.yaml"
    )

    test_dataset = CarsDataset(test_files, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False
    )
    model = get_model_by_name(
        config["trainer"]["model_name"], n_classes=n_classes, config=config
    ).to(device)

    probs = predict(model, test_loader, device)
    preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
    test_filenames = [path.name for path in test_dataset.files]
    targets = [path.parent.name for path in test_dataset.files]
    result = pd.DataFrame({"Id": test_filenames, "Preds": preds, "Target": targets})

    result.sort_values(by="Id", key=natsort_keygen(), inplace=True)

    logger.info("Submission file is saved to submission.csv")
    result.to_csv("submission.csv", index=False)

    logger.info(f"F1 score: {f1_score(preds, targets, average='macro')}")
    logger.info(f"Accuracy score: {accuracy_score(preds, targets)}")
