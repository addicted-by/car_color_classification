import pickle

import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


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
