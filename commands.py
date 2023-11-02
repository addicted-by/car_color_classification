import math
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import fire
import mlflow
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from natsort import natsort_keygen
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from car_color_classification.logger import setup_custom_logger

# from car_color_classification.models import get_model_by_name
from car_color_classification.models import CustomModel
from car_color_classification.utils import get_git_commit_id
from car_color_classification.utils.datasets import CarsDataset
from car_color_classification.utils.load_data import load_data


sys.path.append("./car_color_classification")


logger = setup_custom_logger(__name__)


def cosine_scheduler(optimizer, initial_lr, num_epochs, num_cycles=0.5):
    """
    Cosine learning rate scheduler.

    Args:
        optimizer: The optimizer for which to adjust the learning rate.
        initial_lr (float): The initial learning rate.
        num_epochs (int): The total number of training epochs.
        num_cycles (float): The number of cosine cycles within the training.

    Returns:
        LambdaLR: The PyTorch learning rate scheduler.
    """

    def lr_lambda(epoch):
        """
        Calculate the learning rate multiplier at each epoch.

        Args:
            epoch (int): The current epoch.

        Returns:
            float: The learning rate multiplier for the current epoch.
        """
        cycle = math.floor(1 + epoch / float(num_epochs) * num_cycles)
        x = abs(epoch / float(num_epochs) * num_cycles - cycle + 0.5)
        lr_multiplier = 0.5 * (math.cos(math.pi * x) + 1)
        return lr_multiplier

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    def __init__(self, config, n_classes):
        self.config = config
        self.trainer_config = self.config["trainer"]
        self.val_config = self.config["validation_intermediate"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = n_classes
        self.iterations = 1  # add obtaining from the loaded model
        self._init_train_logger()
        self._init_model()
        self._init_optimizer()
        self._init_criterion()
        self._init_tb_logger()

        self.history = defaultdict(list)

    def _init_train_logger(self):
        pass

    def _init_model(self):
        self.model = CustomModel.get_model_by_name(
            self.config.model.model_name, self.n_classes, self.config
        ).to(self.device)

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            [param for param in self.model.parameters() if param.requires_grad],
            lr=1e-4,
        )

        if self.trainer_config["lr_scheduler"] == "cosine":
            self.scheduler = cosine_scheduler(
                self.optimizer,
                initial_lr=self.trainer_config["max_lr"],
                num_epochs=self.trainer_config["n_epochs"],
                num_cycles=50,
            )

        elif self.trainer_config["lr_scheduler"] == "cyclic":
            self.scheduler = lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.trainer_config["min_lr"],
                max_lr=self.trainer_config["max_lr"],
                step_size_up=5,
                mode="exp_range",
                gamma=0.85,
                last_epoch=-1,
            )
        elif self.trainer_config["lr_scheduler"] == "step":
            self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        elif self.trainer_config["lr_scheduler"] == "no":
            self.scheduler = None
        else:
            raise NotImplementedError("Scheduler is not implemented yet.")

    def _init_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def _init_tb_logger(self):
        self.exp = datetime.now().strftime("%m%d%Y.%H%M%S")
        logdir = f"./tb_logs/{self.config.model.model_name}/{self.exp}"
        logger.info(f"Tensorboard dir: {logdir}")
        self.tb_logger = SummaryWriter(log_dir=logdir)

    def fit(self, train_loader, val_loader):
        path2save = f"ckpts/{self.config.model.model_name}"
        os.makedirs(path2save, exist_ok=True)
        for data, target in train_loader:
            break

        mlflow.models.infer_signature(data.numpy(), target.numpy())

        for epoch in range(self.trainer_config["n_epochs"]):
            self._train_epoch(train_loader, epoch)
            if self.val_config["validate"] and epoch % self.val_config["interval"] == 0:
                self._validate(val_loader, epoch)

            if (epoch + 1) % self.trainer_config["save_interval"] == 0:
                ckpt_name = (
                    f"model_{self.config.model.model_name}" + f"_{self.exp}_{epoch+1}.pth"
                )
                logger.info(f"SAVING MODEL EPOCH {epoch+1} --> {ckpt_name}")
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path2save, ckpt_name),
                )

        mlflow.pytorch.log_model(self.model, path2save)

    def _train_epoch(self, loader, epoch):
        self.model.train()
        pbar = tqdm(loader, unit="batch", total=len(loader), desc=f"Epoch: {epoch + 1}")

        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = self.criterion(preds, target)

            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (preds.argmax(dim=1) == target).sum().item()
            processed_data += batch_size

            if batch_idx % self.trainer_config["log_interval"] == 0:
                accuracy = 100 * running_corrects / processed_data
                self.history["lr"].append(self.optimizer.param_groups[-1]["lr"])
                self.history["train_loss"].append(loss.item())
                self.history["train_accuracy"].append(accuracy)
                self.tb_logger.add_scalar("Loss/train", loss.item(), self.iterations)
                self.tb_logger.add_scalar("Accuracy/train", accuracy, self.iterations)
                self.tb_logger.add_scalar("lr", self.history["lr"][-1], self.iterations)
                mlflow.log_metric("train_loss", loss.item(), self.iterations)
                mlflow.log_metric("train_accuracy", accuracy, self.iterations)
                mlflow.log_metric("lr", self.history["lr"][-1], self.iterations)

                pbar.set_postfix(
                    loss=running_loss / processed_data,
                    accuracy=accuracy,
                    lr=self.optimizer.param_groups[0]["lr"],
                )

            self.iterations += 1

    def _validate(self, loader, epoch):
        self.model.eval()
        pbar = tqdm(
            loader,
            total=len(loader),
            unit="batch",
            desc=f"Validation epoch: {epoch + 1}",
        )

        running_loss = 0.0
        running_corrects = 0
        processed_size = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            with torch.no_grad():
                output = self.model(data)
                loss = self.criterion(output, target)
                preds = torch.argmax(output, dim=1)

            running_loss += loss.item() * target.size(0)
            running_corrects += (preds == target.data).sum().item()
            processed_size += target.size(0)

            accuracy = 100 * running_corrects / processed_size
            pbar.set_postfix(loss=running_loss / processed_size, accuracy=accuracy)

        self.history["val_loss"].append(running_loss / processed_size)
        self.history["val_accuracy"].append(running_corrects / processed_size)
        self.tb_logger.add_scalar(
            "Loss/val", self.history["val_loss"][-1], self.iterations
        )
        self.tb_logger.add_scalar(
            "Accuracy/val", self.history["val_accuracy"][-1], self.iterations
        )
        mlflow.log_metric("val_loss", self.history["val_loss"][-1], self.iterations)
        mlflow.log_metric(
            "val_accuracy", self.history["val_accuracy"][-1], self.iterations
        )


def train(
    cfg="./configs/base/default_arguments.yaml",
    default_arguments: str = "./configs/base/default_arguments.yaml",
):
    cfg = Path(cfg)
    TRAIN_DIR, TEST_DIR = load_data()
    with initialize(
        version_base=None,
        config_path=str(cfg.parent),
        job_name="car_color_classification",
    ):
        config = compose(config_name=cfg.stem)

    if default_arguments:
        default_config = OmegaConf.load(default_arguments)
        config = OmegaConf.merge(default_config, config)

    train_val_files = sorted(list(Path(TRAIN_DIR).rglob("*.jpg")))
    train_val_labels = [path.parent.name for path in train_val_files]

    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )

    n_classes = len(np.unique(train_val_labels))

    trainer = Trainer(config=config, n_classes=n_classes)

    train_dataset = CarsDataset(train_files, mode="train")
    val_dataset = CarsDataset(val_files, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False
    )

    logger.info(config.mlflow.get("tracking_uri"))
    mlflow.set_tracking_uri(config.mlflow.get("tracking_uri", "http://127.0.0.1:5000"))
    os.environ["MLFLOW_ARTIFACT_ROOT"] = config.mlflow.artifact_root
    Path(config.mlflow.artifact_root).mkdir(parents=True, exist_ok=True)

    exp_id = mlflow.set_experiment(f"training-{config.model.model_name}").experiment_id
    with mlflow.start_run(experiment_id=exp_id, run_name=trainer.exp):
        mlflow.log_params(config)
        mlflow.log_param("commit id", get_git_commit_id())
        trainer.fit(train_loader, val_loader)


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


def infer(
    cfg="./configs/inference.yaml", default="./configs/base/default_arguments.yaml"
):
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")

    cfg = Path(cfg)
    TRAIN_DIR, TEST_DIR = load_data()
    with initialize(
        version_base=None,
        config_path=str(cfg.parent),
        job_name="car_color_classification",
    ):
        config = compose(config_name=cfg.stem)

    if default:
        default_config = OmegaConf.load(default)
        config = OmegaConf.merge(default_config, config)

    with open("label_encoder.pkl", "rb") as le_encoder_file:
        label_encoder = pickle.load(le_encoder_file)

    test_files = sorted(list(Path(TEST_DIR).rglob("*.jpg")))
    print(len(test_files))

    n_classes = len(label_encoder.classes_)

    test_dataset = CarsDataset(test_files, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=config.trainer.batch_size, shuffle=False
    )
    model = CustomModel.get_model_by_name(
        config.model.model_name, n_classes=n_classes, cfg=config
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


def run_server():
    pass


if __name__ == "__main__":
    fire.Fire({"train": train, "infer": infer, "run_server": run_server})
