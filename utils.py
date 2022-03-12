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

def load_config(config_path):
    """" Loads the config file into a dictionary. """
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def load_model(**model_kwargs):
    model = model_kwargs['model']
    device = model_kwargs['device']
    model_path = model_kwargs['model_path']
    is_ssl = model_kwargs['ssl']
    """Loads PyTorch model along with statedict(if applicable) to device"""
    if is_ssl:
        pass
    else:
        model = getattr(importlib.import_module("models.{}".format(model)), model)(**model_kwargs)
        model.to(device)
        if model_path:
            model.load_state_dict(torch.load(model_path))
    return model

def load_opt_loss(model, train_config):
    """Fetches optimiser and loss fn params from config and loads"""
    opt_params = train_config['optimizer']
    loss_params = train_config['loss_fn']
    loss_kwargs = {k:loss_params[k] for k in loss_params if k!='name'}
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