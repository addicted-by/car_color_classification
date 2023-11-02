import sys
from pathlib import Path
from typing import Dict

from hydra import compose, initialize
from omegaconf import OmegaConf

from car_color_classification.logger import setup_custom_logger


# from yaml import dump, load


# try:
#     from yaml import CLoader as Loader
# except ImportError:
#     from yaml import Loader


logger = setup_custom_logger(__name__)

sys.path.append("../")


def merge_dicts(dict1, dict2):
    merged = {**dict1, **dict2}
    for key in set(dict1) & set(dict2):
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merged[key] = merge_dicts(dict1[key], dict2[key])
    return merged


def load_config(
    path: Path, default_arguments: str = "./configs/base/default_arguments.yaml"
) -> Dict:
    print(path.parent)
    print(path.stem)
    with initialize(version_base=None, config_path=str(path.parent), job_name="test_app"):
        cfg = compose(config_name=path.stem)

    if default_arguments:
        default_config = OmegaConf.load(default_arguments)
        cfg = OmegaConf.merge(default_config, cfg)

    print(OmegaConf.to_yaml(cfg))
    return cfg


if __name__ == "__main__":
    load_config(
        "../../configs/base/default_arguments.yaml",
        "../../configs/base/default_arguments.yaml",
    )
    load_config(
        "../../configs/custom_config.yaml", "../../configs/base/default_arguments.yaml"
    )
    load_config(
        "../../configs/inference.yaml", "../../configs/base/default_arguments.yaml"
    )
