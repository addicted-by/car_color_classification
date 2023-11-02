# from .config import load_config
# from .datasets import CarsDataset
# from .load_data import load_data


import subprocess


def get_git_commit_id():
    try:
        # Use subprocess to execute git commands
        git_commit_id = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        )
        return git_commit_id
    except Exception as e:
        print("Failed to get the Git commit ID:", e)
        return None
