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
from external_lib.SSL.models import SIMCLR, SIMSIAM
from models.LinearEval import SSLEvaluator, SSLEvaluatorOneLayer
from models.SSLClassifier import ClassifierModel


def load_config(config_path):
    """" Loads the config file into a dictionary. """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_model(**model_kwargs):
    model = model_kwargs['model']
    device = model_kwargs['device']
    model_path = model_kwargs['model_path']
    ssl_config = model_kwargs.get('ssl', {})
    """Loads PyTorch model along with statedict(if applicable) to device"""
    if ssl_config:
        model_path = model_kwargs['encoder']['encoder_path']
        if ssl_config['encoder']['encoder_type'] == 'SIMCLR':
            encoder = SIMCLR.SIMCLR.load_from_checkpoint(model_path, DATA_PATH = GConst.UNLABELLED_DIR).encoder
        elif ssl_config['encoder']['encoder_type'] == 'SIMSIAM':
            encoder = SIMSIAM.SIMSIAM.load_from_checkpoint(model_path, DATA_PATH = GConst.UNLABELLED_DIR).encoder
        
        if ssl_config["classifier"]["classifier_type"] == "SSLEvaluator":
            linear_model = SSLEvaluator(
                n_input = ssl_config["encoder"]["e_embedding_size"],
                n_classes = ssl_config["classifier"]["c_num_classes"],
                p = ssl_config["classifier"]["c_dropout"],
                n_hidden = ssl_config["classifier"]["c_hidden_dim"],
            )
        
        elif ssl_config["classifier"]["classifier_type"] == "SSLEvaluatorOneLayer":
            linear_model = SSLEvaluatorOneLayer(
                n_input = ssl_config["encoder"]["e_embedding_size"],
                n_classes = ssl_config["classifier"]["c_num_classes"],
                p = ssl_config["classifier"]["c_dropout"],
                n_hidden = ssl_config["classifier"]["c_hidden_dim"],
            )

        model = ClassifierModel(device, encoder, linear_model)

        

    else:
        model = getattr(importlib.import_module("models.{}".format(model)), model)(**model_kwargs)
        model.to(device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
    return model


def load_opt_loss(model, config, is_ssl = False):
    """Fetches optimiser and loss fn params from config and loads"""
    opt_params = config['train']['optimizer']
    loss_params = config['train']['loss_fn']
    ssl_config = config['model'].get('ssl', {})
    loss_kwargs = {k:loss_params[k] for k in loss_params if k!='name'}
    if ssl_config:
        encoder_lr = ssl_config['encoder']['e_lr'] if ssl_config['encoder']['train_encoder'] else 0
        optimizer = getattr(optim, opt_params['name'])(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.linear_model.parameters(), "lr": ssl_config['classifier']['c_lr']},
            ],
            **opt_params.get('config', {})
        )
    else:
        optimizer = getattr(optim, opt_params['name'])(
                    model.parameters(), **opt_params.get('config', {}))

    loss_fn = getattr(nn, loss_params['name'])(**loss_kwargs)

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

def copy_data(paths, folder):
    for image in tqdm(paths):
        shutil.copy(image, os.path.join(folder, image))
    print('Data Copied to {}'.format(folder))

def get_num_files(folder):
    if folder == "positive":
        return len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR,'positive'))))
    elif folder == 'negative':
        return len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR,'negative'))))
    elif folder == 'unlabelled':
        return len(list(paths.list_images(GConst.UNLABELLED_DIR)))
    elif folder == 'eval_pos':
        return len(list(paths.list_images(os.path.join(GConst.EVAL_DIR,'positive'))))
    elif folder == 'eval_neg':
        return len(list(paths.list_images(os.path.join(GConst.EVAL_DIR,'negative'))))
    elif folder == 'labelled':
        return len(list(paths.list_images(GConst.LABELLED_DIR)))
    elif folder == 'eval_labelled':
        return len(list(paths.list_images(GConst.EVAL_DIR)))
        
def annotate_data(paths, folder):
    if folder == "positive":
        copy_data(paths, os.path.join(GConst.LABELLED_DIR,'positive'))
    elif folder == 'negative':
        copy_data(paths, os.path.join(GConst.LABELLED_DIR,'negative'))
    elif folder == 'unlabelled':
        copy_data(paths, GConst.UNLABELLED_DIR)
    elif folder == 'eval_pos':
        copy_data(paths, os.path.join(GConst.EVAL_DIR,'positive'))
    elif folder == 'eval_neg':
        copy_data(paths, os.path.join(GConst.EVAL_DIR,'negative'))