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
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import importlib
import warnings
from operator import itemgetter
warnings.filterwarnings("ignore")

from data.resisc import resisc_annotate, resisc_download


def main(dataset, model_class, model_path csv_path=None):
    model = importlib.import_module("models.{}".format(model_class))
    model.to('cuda')
    if model_path:
        model.load_state_dict(torch.load(model_path))




if "__name__" == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,help='resisc/custom dataset')
    parser.add_argument('--csv_path', type=str, default=None, help='csv path if csv is chosen')
    parser.add_argument('--model', type=str, default=None, help='model class name')
    args = parser.parse_args()
