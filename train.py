import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from car_color_classification.models import get_model_by_name
from car_color_classification.utils.args import parse_arguments
from car_color_classification.utils.config import load_config
from car_color_classification.utils.datasets import CarsDataset
from car_color_classification.utils.load_data import load_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


sys.path.append("./car_color_classification")


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
        self.model = get_model_by_name(
            self.trainer_config["model_name"], self.n_classes, self.config
        ).to(self.device)

    def _init_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            list(self.model.layer3.parameters()) + list(self.model.layer4.parameters()),
            lr=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

    def _init_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def _init_tb_logger(self):
        experiment = datetime.now().strftime("%m%d%Y.%H%M%S")
        logdir = f"./tb_logs/{self.trainer_config['model_name']}/{experiment}"
        self.tb_logger = SummaryWriter(log_dir=logdir)

    def fit(self, train_loader, val_loader):
        exp = datetime.now().strftime("%m%d%Y.%H%M%S")
        path2save = f"./ckpts/{self.trainer_config['model_name']}/"
        os.makedirs(path2save, exist_ok=True)

        for epoch in range(self.trainer_config["n_epochs"]):
            self._train_epoch(train_loader, epoch)
            if self.val_config["validate"] and epoch % self.val_config["interval"] == 0:
                self._validate(val_loader, epoch)

            if (epoch + 1) % self.trainer_config["save_interval"] == 0:
                print("SAVING MODEL")
                ckpt_name = (
                    f"model_{self.trainer_config['model_name']}" + f"_{exp}_{epoch+1}.pth"
                )
                torch.save(
                    self.model.state_dict(),
                    os.path.join(path2save, ckpt_name),
                )

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

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (preds.argmax(dim=1) == target).sum().item()
            processed_data += batch_size

            if batch_idx % self.trainer_config["log_interval"] == 0:
                accuracy = 100 * running_corrects / processed_data
                self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
                self.history["train_loss"].append(loss.item())
                self.history["train_accuracy"].append(accuracy)
                self.tb_logger.add_scalar("Loss/train", loss.item(), self.iterations)
                self.tb_logger.add_scalar("Accuracy/train", accuracy, self.iterations)
                self.tb_logger.add_scalar("lr", self.history["lr"][-1], self.iterations)

                pbar.set_postfix(loss=running_loss / processed_data, accuracy=accuracy)

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


if __name__ == "__main__":
    args = parse_arguments(default="./configs/base/default_arguments.yaml")

    TRAIN_DIR, TEST_DIR = load_data()
    train_val_files = sorted(list(Path(TRAIN_DIR).rglob("*.jpg")))
    test_files = sorted(list(Path(TEST_DIR).rglob("*.jpg")))

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )

    n_classes = len(np.unique(train_val_labels))
    config = load_config(
        args.config, default_arguments="./configs/base/default_arguments.yaml"
    )
    trainer = Trainer(config=config, n_classes=n_classes)

    train_dataset = CarsDataset(train_files, mode="train")
    val_dataset = CarsDataset(val_files, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False
    )

    trainer.fit(train_loader, val_loader)