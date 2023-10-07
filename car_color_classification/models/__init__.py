import os
import warnings
from typing import Dict

import torch
import torchvision
from natsort import natsorted


warnings.filterwarnings(action="ignore", category=UserWarning)
# from .resnet101 import ResNet101


def get_model_by_name(model_name: str, n_classes: int, config: Dict):
    if model_name == "resnet101":
        pretrained = config["trainer"]["pretrained"]
        model = torchvision.models.resnet101(pretrained=pretrained)

        if config["trainer"]["freeze_layers"]:
            for num, child in enumerate(model.children()):
                if num < 6:
                    for param in child.parameters():
                        param.requires_grad = False

        model.fc = torch.nn.Linear(2048, n_classes)
        model.add_module("SoftMax", torch.nn.Softmax(dim=-1))
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=pretrained)
        for num, child in enumerate(model.children()):
            if num < 6:
                for param in child.parameters():
                    param.requires_grad = False

        model.fc = torch.nn.Linear(2048, n_classes)
        model.add_module("SoftMax", torch.nn.Softmax(dim=-1))

    else:
        raise ValueError(f"Cannot process this model name: {model_name}!")

    if config["trainer"]["ckpt_load"]:
        ckpt = config["trainer"]["ckpt_load"]
        if ckpt == "last":
            ckpts_path = os.path.join(
                config["trainer"]["ckpt_dir"], config["trainer"]["model_name"]
            )
            last_ckpt = natsorted(os.listdir(ckpts_path))[-1]
            ckpt = os.path.join(ckpts_path, last_ckpt)

        print(f"Loading checkpoint {ckpt}")
        err_msg = f"Checkpoint {ckpt} does not exists. Please check!"
        assert os.path.exists(ckpt), err_msg
        model.load_state_dict(torch.load(ckpt))

    return model
