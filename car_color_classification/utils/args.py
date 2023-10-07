import argparse


def parse_arguments(default):
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument(
        "--config", help="Path to the configuration YAML file", default=default
    )
    return parser.parse_args()
