from json import encoder
import yaml
import torch
import importlib
from torch import optim
import torch.nn as nn
import os
import shutil
from tqdm import tqdm
import global_constants as GConst
from imutils import paths
import sys

sys.path.append("external_lib/SSL/")

from external_lib.SSL.models import SIMCLR, SIMSIAM
from models.LinearEval import SSLEvaluator, SSLEvaluatorOneLayer
from models.SSLClassifier import ClassifierModel


def load_config(config_path):
    """ " Loads the config file into a dictionary."""
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def load_model(**model_kwargs):
    device = model_kwargs["device"]

    if device != "cuda":
        raise NotImplementedError(
            "Currently supporting only cuda backends, please change device in the config to cuda."
        )

    ssl_config = model_kwargs.get("ssl", {})
    """Loads PyTorch model along with statedict(if applicable) to device"""
    if ssl_config:
        model_path = ssl_config["encoder"]["encoder_path"]
        if ssl_config["encoder"]["encoder_type"] == "SIMCLR":
            encoder = SIMCLR.SIMCLR.load_from_checkpoint(
                model_path, DATA_PATH=GConst.UNLABELLED_DIR
            ).encoder
        elif ssl_config["encoder"]["encoder_type"] == "SIMSIAM":
            encoder = SIMSIAM.SIMSIAM.load_from_checkpoint(
                model_path, DATA_PATH=GConst.UNLABELLED_DIR
            ).encoder

        if ssl_config["classifier"]["classifier_type"] == "SSLEvaluator":
            linear_model = SSLEvaluator(
                n_input=ssl_config["encoder"]["e_embedding_size"],
                n_classes=ssl_config["classifier"]["c_num_classes"],
                p=ssl_config["classifier"]["c_dropout"],
                n_hidden=ssl_config["classifier"]["c_hidden_dim"],
            )

        elif ssl_config["classifier"]["classifier_type"] == "SSLEvaluatorOneLayer":
            linear_model = SSLEvaluatorOneLayer(
                n_input=ssl_config["encoder"]["e_embedding_size"],
                n_classes=ssl_config["classifier"]["c_num_classes"],
                p=ssl_config["classifier"]["c_dropout"],
                n_hidden=ssl_config["classifier"]["c_hidden_dim"],
            )

        model = ClassifierModel(device, encoder, linear_model)

        if not ssl_config["encoder"]["train_encoder"]:
            model.freeze_encoder()
        else:
            model.unfreeze_encoder()

    else:
        model = model_kwargs["model"]
        model_path = model_kwargs["model_path"]
        model = getattr(importlib.import_module("models.{}".format(model)), model)(
            **model_kwargs
        )
        model.to(device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
    return model


def load_opt_loss(model, config, is_ssl=False):
    """Fetches optimiser and loss fn params from config and loads"""
    opt_params = config["train"]["optimizer"]
    loss_params = config["train"]["loss_fn"]
    ssl_config = config["model"].get("ssl", {})
    loss_kwargs = {k: loss_params[k] for k in loss_params if k != "name"}
    if ssl_config:
        encoder_lr = (
            ssl_config["encoder"]["e_lr"]
            if ssl_config["encoder"]["train_encoder"]
            else 0
        )
        optimizer = getattr(optim, opt_params["name"])(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {
                    "params": model.linear_model.parameters(),
                    "lr": ssl_config["classifier"]["c_lr"],
                },
            ],
            **opt_params.get("config", {}),
        )
    else:
        optimizer = getattr(optim, opt_params["name"])(
            model.parameters(), **opt_params.get("config", {})
        )

    loss_fn = getattr(nn, loss_params["name"])(**loss_kwargs)

    return optimizer, loss_fn


def get_num_files(path):
    return len(list(paths.list_images(path)))


def initialise_data_dir(config):
    if os.path.exists("Dataset/Labeled"):
        shutil.rmtree("Dataset/")
    os.makedirs("Dataset/Labeled/")
    for i in config["data"]["classes"]:
        os.makedirs(f"Dataset/Labeled/{i}")


    if os.path.exists("Dataset/Val"):
        shutil.rmtree("Dataset/")
    os.makedirs("Dataset/Val/")
    for i in config["data"]["classes"]:
        os.makedirs(f"Dataset/Val/{i}")
    
    if os.path.exists("checkpoints/"):
        shutil.rmtree("checkpoints/")
    os.makedirs("checkpoints/")

    if os.path.exists("logs/"):
        shutil.rmtree("logs/")
    os.makedirs('logs/')


def copy_data(paths, folder):
    for image in tqdm(paths):
        shutil.copy(image, folder)
    print("Data Copied to {}".format(folder))
