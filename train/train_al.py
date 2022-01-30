from distutils.command.config import config
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
import global_constants as GConst
warnings.filterwarnings("ignore")

from train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched


    # print("Total Labeled Data: Positive {} Negative {}".format(len(list(paths.list_images('/content/Dataset/Labelled/positive'))),len(list(paths.list_images('/content/Dataset/Labelled/negative')))))



plt.plot(graph_logs['len_data'],graph_logs['val_f1'])
plt.scatter(graph_logs['len_data'],graph_logs['val_f1'])
plt.xlabel('Amount of Training Data Used')
plt.ylabel('Validation F1 Score')
plt.title('F1 Score across Iterations against Amount of Training Data')
plt.show()