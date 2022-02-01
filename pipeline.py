from distutils.command.config import config
import gc
import os 
import imutils
from random import shuffle 
from imutils import paths 
import matplotlib.pyplot as plt
from PIL import Image
import pathlib
from pathlib import Path
import argparse
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import importlib
import warnings
from operator import itemgetter
from data.custom_datasets import AL_Dataset, RESISC_Eval
import global_constants as GConst
warnings.filterwarnings("ignore")

from utils import (load_config, load_model, load_opt_loss, initialise_data_dir)
from data import resisc
from train.train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched


class Pipeline:

    def __init__(self, config_path) -> None:
        self.config = load_config(config_path)
        model = self.config['model']['model']
        model_kwargs = self.config['model'].get('model_params', {})
        model_path = self.config['model'].get('model_path', None)
        device = self.config['model']['device']
        self.model = load_model(model, model_path, device, **model_kwargs)
        self.optim, self.loss = load_opt_loss(self.model, self.config['train'])
        self.already_labelled = list()
        self.transform = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))])

        initialise_data_dir()

    def main(self):
        config = self.config
        if config['data']['dataset'] == 'resisc':
            positive_class = config['data']['positive_class']
            resisc.download_and_prepare()
            #Initialising data by annotating labelled and eval
            unlabelled_images = list(paths.list_images(GConst.UNLABELLED_DIR))
            self.already_labelled = resisc.resisc_annotate(unlabelled_images, 100, self.already_labelled, positive_class, labelled_dir=GConst.EVAL_DIR, val=True) 
            self.already_labelled = resisc.resisc_annotate(unlabelled_images, 50, self.already_labelled, positive_class, labelled_dir=GConst.LABELLED_DIR)
            print("Total Eval Data: Positive {} Negative {}".format(len(list(paths.list_images(os.path.join(GConst.EVAL_DIR, 'positive')))),len(list(paths.list_images(os.path.join(GConst.EVAL_DIR, 'negative'))))))
            print("Total Labeled Data: Positive {} Negative {}".format(len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR, 'positive')))),len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR, 'negative'))))))

            #Train 
            eval_dataset = RESISC_Eval(GConst.UNLABELLED_DIR, positive_class)
            val_dataset = ImageFolder(GConst.EVAL_DIR, transform = self.transform)

            train_config = config['train']
            train_kwargs = dict(epochs = train_config['epochs'],
                                opt = self.optim, 
                                loss_fn = self.loss, 
                                batch_size = train_config['batch_size'],
                                )
            al_config = config['active_learner']
            al_kwargs = dict(
                            eval_dataset = eval_dataset, 
                            val_dataset=  val_dataset, 
                            strategy = al_config['strategy'],
                            positive_class = positive_class,
                            num_iters = al_config['iterations'],
                            num_labelled = al_config['num_labelled']
                            )
            
            
            logs = self.train_al(self.model, self.already_labelled, unlabelled_images, train_kwargs, **al_kwargs)

    def train_al(self, model, already_labelled, unlabelled_images, train_kwargs, **al_kwargs):
        iter = 0
        eval_dataset = al_kwargs['eval_dataset']
        val_dataset = al_kwargs['val_dataset']
        num_iters = al_kwargs['num_iters']
        positive_class = al_kwargs['positive_class']
        
        logs = {'ckpt_path' : [],
                'graph_logs': []}

        while iter < num_iters:
            print(f'-------------------{iter +1}----------------------')
            iter+=1
            ckpt_path, graph_logs = train_model_vanilla(self.model, GConst.LABELLED_DIR, eval_dataset, val_dataset, **train_kwargs)
            logs['ckpt_path'].append(ckpt_path)
            logs['graph_logs'].append(graph_logs)
            low_confs = get_low_conf_unlabeled_batched(model, unlabelled_images, **al_kwargs)
            print("Images selected from: ",len(low_confs))
            for image in low_confs:
                if image not in already_labelled:
                    self.already_labelled.append(image)
                if image.split('/')[-1].split('_')[0] == positive_class:
                    shutil.copy(image, os.path.join(GConst.LABELLED_DIR,'positive',image.split('/')[-1]))
                else:
                    shutil.copy(image, os.path.join(GConst.LABELLED_DIR,'negative',image.split('/')[-1]))
            print("Total Labeled Data: Positive {} Negative {}".format(len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR, 'positive')))),len(list(paths.list_images(os.path.join(GConst.LABELLED_DIR, 'negative'))))))

        return logs





