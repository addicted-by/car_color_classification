import pickle
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms

from car_color_classification.utils.load_data import load_data


DATA_MODES = ["train", "val", "test"]
RESCALE_SIZE = 224
DEVICE = torch.device("cuda")


class CarsDataset(Dataset):
    """
    Images dataset which loads them from the directories and make their scaling
    and converting to tensors.
    """

    def __init__(self, files, mode, transform=None):
        super().__init__()
        self.files = sorted(files)
        self.mode = mode
        self.transform = (
            transform
            if transform
            else transforms.Compose(
                [
                    transforms.Resize((RESCALE_SIZE, RESCALE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        )

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != "test":
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open("label_encoder.pkl", "wb") as le_dump_file:
                pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self.transform(x)
        if self.mode == "test":
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        val_size: float,
        dataloader_num_wokers: int,
        batch_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.val_size = val_size
        self.dataloader_num_wokers = dataloader_num_wokers
        self.batch_size = batch_size

    def prepare_data(self):
        self.TRAIN_DIR, self.TEST_DIR = load_data()

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            train_val_files = sorted(list(Path(self.TRAIN_DIR).rglob("*.jpg")))
            train_val_labels = [path.parent.name for path in train_val_files]

            train_files, val_files = train_test_split(
                train_val_files, test_size=self.val_size, stratify=train_val_labels
            )
            self.train_dataset = CarsDataset(train_files, mode="train")
            self.val_dataset = CarsDataset(val_files, mode="val")

        if stage == "test" or stage == "predict":
            test_files = sorted(list(Path(self.TEST_DIR).rglob("*.jpg")))
            self.test_dataset = CarsDataset(test_files, mode="test")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_wokers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_wokers,
        )
