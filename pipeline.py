from ast import Index
from copy import copy
from distutils.command.config import config
import gc
import sys
import os 
import imutils
from random import shuffle 
import pandas as pd
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
import global_constants as GConst
warnings.filterwarnings("ignore")
sys.path.append('{}/external_lib/SSL/'.format(os.getcwd()))

from utils import (copy_data, load_config, load_model, load_opt_loss, initialise_data_dir, annotate_data, get_num_files)
from data import resisc
from train.train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched
from data.custom_datasets import AL_Dataset, RESISC_Eval
from data.swipe_labeler import SwipeLabeller
from data.indexer import Indexer

def adhoc_copy(unlabelled_paths):
    imgs = unlabelled_paths['image_paths'].values[:4]
    for i in range(len(imgs)):
        if i %2 == 0:
            shutil.copy(imgs[i], os.path.join(GConst.LABELLED_DIR, 'negative'))
        else:
            shutil.copy(imgs[i], os.path.join(GConst.EVAL_DIR, 'negative'))
    print('ADHOC DONE : ', len(imgs))
class Pipeline:

    def __init__(self, config_path) -> None:
        self.config = load_config(config_path)
        initialise_data_dir()

        model_kwargs = self.config['model']
        self.model = load_model(**model_kwargs)
        self.optim, self.loss = load_opt_loss(self.model, self.config)
        self.already_labelled = list()
        self.transform = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))])
        self.sl = SwipeLabeller(self.config)


    def main(self):
        config = self.config
        if config['data']['dataset'] == 'resisc':
            positive_class = config['data']['positive_class']
            resisc.download_and_prepare()
            #Initialising data by annotating labelled and eval
            unlabelled_images = list(paths.list_images(GConst.UNLABELLED_DIR))
            self.already_labelled = resisc.resisc_annotate(unlabelled_images, 100, self.already_labelled, positive_class, labelled_dir=GConst.EVAL_DIR, val=True) 
            self.already_labelled = resisc.resisc_annotate(unlabelled_images, 50, self.already_labelled, positive_class, labelled_dir=GConst.LABELLED_DIR)
            print("Total Eval Data: Positive {} Negative {}".format(get_num_files("eval_pos"),get_num_files('eval_neg')))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files("positive"),get_num_files('negative')))

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
                            num_labelled = al_config['num_labelled'],
                            limit  = al_config['limit']
                            )
            logs = self.train_al_resisc(self.model, self.already_labelled, unlabelled_images, train_kwargs, **al_kwargs)
        
        elif config['data']['dataset'] == 'csv':

            self.df = pd.read_csv(config['data']['path'])
            df = self.df.copy()
            query_image = df[df['status'] == 'query'][GConst.IMAGE_PATH_COL].values
            unlabelled_paths = df[df['status'] != 'query']
            unlabelled_paths_lis = unlabelled_paths[GConst.IMAGE_PATH_COL].values
            num_labelled = config['active_learner']['num_labelled']
            self.preindex = self.config['active_learner']['preindex']
            if self.preindex:
                self.index = Indexer(unlabelled_paths_lis, self.model, img_size=224, 
                                     index_path = None)
            
            if len(query_image) > 1:
                split_ratio = int(0.9 * len(query_image)) #TODO make this an arg
                annotate_data(query_image[split_ratio:], 'eval_pos')
                annotate_data(query_image[:split_ratio], 'positive')
            else:
                annotate_data(query_image, 'positive')
                annotate_data(query_image, 'eval_pos')
            
            if self.preindex: 
                #FAISS Fetch
                similar_imgs = self.index.process_image(query_image[0], n_neighbors=num_labelled *2) #hardcoding sending only the first image here from query images
                train_init = similar_imgs[:num_labelled]
                val_init = similar_imgs[num_labelled:]
                self.sl.label(train_init, is_eval=False)
                self.sl.label(val_init, is_eval = True)
            else:
                random_init_imgs = unlabelled_paths.sample(num_labelled * 2)[GConst.IMAGE_PATH_COL].values
                train_init = random_init_imgs[:num_labelled]
                val_init = random_init_imgs[num_labelled:]

                self.sl.label(train_init, is_eval=False)
                self.sl.label(val_init, is_eval = True)

            #swipe_labeler -> label random set of data -> labelled pos/neg. Returns paths labelled
            self.already_labelled.extend(random_init_imgs)
            print("Total annotated valset : {} Positive {} Negative".format(get_num_files("eval_pos"),get_num_files('eval_neg')))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files("positive"),get_num_files('negative')))
            
            train_config = config['train']    
            #data is ready ,start training and AL   
            train_kwargs = dict(epochs = train_config['epochs'],
                                opt = self.optim,
                                loss_fn = self.loss, 
                                batch_size = train_config['batch_size']
                                )
                                
            al_config = config['active_learner']
            al_kwargs = dict(
                            strategy = al_config['strategy'],
                            num_iters = al_config['iterations'],
                            num_labelled = al_config['num_labelled'],
                            limit  = al_config['limit']
                            )

            adhoc_copy(unlabelled_paths)

            logs = self.train_al(self.model, unlabelled_paths_lis, train_kwargs, **al_kwargs)


    def train_al(self, model, unlabelled_images, train_kwargs, **al_kwargs):
        iter = 0
        num_iters = al_kwargs['num_iters']

        logs = {'ckpt_path' : [],
                'graph_logs' : []}
        
        while iter < num_iters:
            print(f'-------------------{iter +1}----------------------')
            iter+=1
            ckpt_path, graph_logs = train_model_vanilla(self.model, GConst.LABELLED_DIR, **train_kwargs)
            logs['ckpt_path'].append(ckpt_path)
            logs['graph_logs'].append(graph_logs)
            low_confs = get_low_conf_unlabeled_batched(model, unlabelled_images, self.already_labelled, **al_kwargs)
            self.sl.label(low_confs, is_eval = False)
            self.already_labelled.extend(low_confs)
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files('positive'), get_num_files('negative')))

        return logs



    def train_al_resisc(self, model, unlabelled_images, train_kwargs, **al_kwargs):
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
            low_confs = get_low_conf_unlabeled_batched(model, unlabelled_images, self.already_labelled, **al_kwargs)
            print("Images selected from: ",len(low_confs))
            for image in low_confs:
                if image not in self.already_labelled:
                    self.already_labelled.append(image)
                if image.split('/')[-1].split('_')[0] == positive_class:
                    shutil.copy(image, os.path.join(GConst.LABELLED_DIR,'positive',image.split('/')[-1]))
                else:
                    shutil.copy(image, os.path.join(GConst.LABELLED_DIR,'negative',image.split('/')[-1]))
            print("Total Labeled Data: Positive {} Negative {}".format(get_num_files('positive'), get_num_files('negative')))

        return logs





