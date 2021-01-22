import sys
sys.path.insert(0, '/content/test_dir')
sys.path.insert(0, '/content/test_dir/SpaceForceDataSearch')
print(sys.path)

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import statistics
from pathlib import Path
import h5py
import scann
import pandas as pd
from argparse import ArgumentParser
import os
import torch
import numpy as np

#Curator Imports
from SpaceForceDataSearch.ssl_dali_distrib import SIMCLR 
from SpaceForceDataSearch.finetuner_dali_distrib import finetuner

def load_checkpoint(MODEL_PATH):
    #expects a checkpoint path, not an encoder
    try:
        model = finetuner.load_from_checkpoint(MODEL_PATH)
        is_classifier = True
    except:
        model = SIMCLR.load_from_checkpoint(MODEL_PATH)
        is_classifier = False
    return model, is_classifier

def nearest_neighbors(embedding_matrix, search_matrix, N): 
    '''
    Logic of nearest_neighbor: find nearest_neighbors to search_matrix
    '''

    #add the reference images
    full_matrix = np.vstack((search_matrix, embedding_matrix))
    if os.path.exists('./temp_embedding_data.h5'):
      os.remove('./temp_embedding_data.h5')

    f = h5py.File('./temp_embedding_data.h5', 'w')
    f.create_dataset("embeddings", data=full_matrix)

    dataset_scann = f['embeddings']
    
    normalized_dataset = dataset_scann / np.linalg.norm(dataset_scann, axis=1)[:, np.newaxis]

    searcher = scann.scann_ops_pybind.builder(normalized_dataset, N+1, "dot_product").tree(num_leaves = int(np.sqrt(len(dataset_scann))), num_leaves_to_search = 10).score_brute_force().build() 

    neighbors, distances = searcher.search_batched(normalized_dataset)
    search_neighbors = neighbors[:search_matrix.shape[0], 1:].flatten().astype(np.int32)
    
    os.remove('./temp_embedding_data.h5')

    #remove duplicates, references to search_matrix, and reindex 
    search_neighbors = pd.unique(search_neighbors) - search_matrix.shape[0]
  
    return search_neighbors[search_neighbors >= 0][:N]

def get_matrix(model, DATA_PATH, only_class = None, input_height = 256, batch_size = 32):
    crop = transforms.RandomResizedCrop(size = input_height, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=2)
    tensor = transforms.ToTensor()
    t = transforms.Compose([crop, tensor])
    dataset = ImageFolder(DATA_PATH, transform=t)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory=True) 

    m_size = model.num_classes if model.is_classifier else model.embedding_size

    embedding_matrix = torch.empty((0, m_size)).cuda()

    for batch in loader:
        with torch.no_grad():
          embeddings = model(batch[0].cuda())
          embedding_matrix = torch.vstack((embedding_matrix, embeddings))

    embedding_matrix = embedding_matrix.cpu().detach().numpy()
    
    if only_class is not None:
        only_idx = np.array(dataset.imgs)[:,1] == str(only_class)
        return embedding_matrix[only_idx], np.array(dataset.imgs)[:,0][only_idx]
    else:
        return embedding_matrix, np.array(dataset.imgs)[:,0]

def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help="path to folders with images")
    parser.add_argument('--to_be_labeled', type=str, help='folder with images to be labeled')
    parser.add_argument("--MODEL_PATH", default=None, type=str, help="classifier checkpoint or SSL checkpoint")
    parser.add_argument("--batch_size", default=128, type=int, help="batch size for SSL")
    parser.add_argument("--candidates", default=200, type=int, help="number of candidates to populate /To_Be_Labeled Folder")
    parser.add_argument("--image_size", default=256, type=int, help="height of square image")
    
    args = parser.parse_args()
    DATA_PATH = args.DATA_PATH
    MODEL_PATH = args.MODEL_PATH
    batch_size = args.batch_size
    N = args.candidates
    input_height = args.image_size
    TO_LABEL = args.to_be_labeled

    model, is_classifier = load_checkpoint(MODEL_PATH)
    model.is_classifier = is_classifier
    model.eval()
    model.cuda()
    embedding_matrix, file_list = get_matrix(model, DATA_PATH, input_height = input_height, batch_size = batch_size)

    if not is_classifier:
        #get indecisives
        idxs = np.argsort(embedding_matrix.std(axis = 1))[:N]
    else:
        #get nearest neighbors idx
        search_matrix, _ = get_matrix(model, '/'.join(TO_LABEL.split('/')[:-1]) + '/Labeled', only_class = '1', input_height = input_height,  batch_size = batch_size)
        idxs = nearest_neighbors(embedding_matrix, search_matrix, N)

    files = file_list[idxs]

    #moves files to the "./To_Be_Labeled" Folder

    Path(TO_LABEL).mkdir(exist_ok=True)

    for f in files:
      os.rename(f, f"{TO_LABEL}/{f.split('/')[-1]}")

if __name__ == '__main__':
    cli_main()
