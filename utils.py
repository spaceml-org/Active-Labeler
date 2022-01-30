import yaml
import torch
import importlib
from torch import optim
import torch.nn as nn
import os 
import shutil

def load_config(config_path):
    """" Loads the config file into a dictionary. """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_model(model, model_path = None, device = 'cuda', **model_kwargs):
    """Loads PyTorch model along with statedict(if applicable) to device"""
    model = getattr(importlib.import_module("models.{}".format(model)), model)(**model_kwargs)
    model.to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    return model

def load_opt_loss(train_config, model):
    """Fetches optimiser and loss fn params from config and loads"""
    opt_params = train_config['optimizer']
    loss_params = train_config['loss_fn']
    optimizer = getattr(optim, opt_params['name'])(
                model.parameters(), **opt_params.get('config', {}))
    loss_fn = getattr(nn, loss_params['name'])

    return optimizer, loss_fn


def initialise_data_dir():
    if os.path.exists('Dataset/Labelled'):
        shutil.rmtree('Dataset/Labelled')
    if os.path.exists('Dataset/Eval'):
        shutil.rmtree('Dataset/Eval')
    
    if os.path.exists('checkpoints/'):
        shutil.rmtree('checkpoints/')      

    os.makedirs('Dataset/Labelled/positive')
    os.makedirs('Dataset/Labelled/negative')
    os.makedirs('Dataset/Eval/positive')
    os.makedirs('Dataset/Eval/negative')
    os.makedirs('checkpoints/')
