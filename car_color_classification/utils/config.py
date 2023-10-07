import os
from typing import Dict

from yaml import dump, load


try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import sys

from car_color_classification.logger import setup_custom_logger


logger = setup_custom_logger(__name__)

sys.path.append("../")


def print_config(config):
    print(dump(config, default_flow_style=False))


def merge_dicts(dict1, dict2):
    merged = {**dict1, **dict2}
    for key in set(dict1) & set(dict2):
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            merged[key] = merge_dicts(dict1[key], dict2[key])
    return merged


def load_config(
    path: str, default_arguments: str = "configs/base/default_arguments.yaml"
) -> Dict:
    assert os.path.exists(path), f"Bad specified path {path}"
    custom_config = load(open(path, "r"), Loader=Loader)
    final_config = custom_config
    # Load default config
    if default_arguments:
        assert os.path.exists(
            default_arguments
        ), f"Can't find base config {default_arguments}"
        default_config = load(open(default_arguments, "r"), Loader=Loader)
        final_config = merge_dicts(default_config, custom_config)

    logger.info("Config file: ")
    print_config(final_config)
    return final_config
