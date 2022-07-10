import os
from utils import load_config
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to config file")
args = parser.parse_args()
config_path = args.config_path

config = load_config(config_path)
dataset_name = config["data"]["dataset_name"]

LABELED_DIR = f"{dataset_name}/labeled"
VAL_DIR = f"{dataset_name}/validation"
TEST_DIR = f"{dataset_name}/test"
UNLABELED_DIR = f"{dataset_name}/unlabeled"


IMAGE_PATH_COL = "image_paths"

DATASET_NAME = f"{dataset_name}"

start_name = config["active_learner"]["strategy"]
diversity_name = config["active_learner"]["diversity_sampling"]
