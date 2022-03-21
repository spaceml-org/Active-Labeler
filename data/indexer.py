from argparse import ArgumentParser
import logging
import pickle
import os

import numpy as np
from tqdm.notebook import tqdm
import PIL.Image as Image
import matplotlib.pyplot as plt
import faiss

from torchvision import transforms
from data.custom_datasets import AL_Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torch
from PIL import Image
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


class Indexer:
    def __init__(self, imgs, model, img_size=224, index_path = None) -> None:

        self.model = model
        self.imgs = imgs
        self.embeddings = get_matrix(self.model, self.imgs, img_size)

        self.images_list = list(imgs)
        
        if index_path:
            self.index = faiss.read_index(index_path)
        else:    
            self.index = index_gen(self.embeddings)

    def process_image(self, img, n_neighbors=5):
        src = get_embedding(self.model, img)
        scores, neighbours = self.index.search(x=src, k=n_neighbors)
        neighbours = neighbours[0]
        result = list(self.images_list[neighbours[i]] for i in range(len(neighbours)))
        return result

def get_matrix(model, DATA_PATH, image_size=224) -> np.ndarray:

    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

    t = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.Lambda(to_tensor)]
    )
    inf_model = model.copy()
    #define output layer
    inf_model.enc.fc = nn.Identity()

    dataset = AL_Dataset(DATA_PATH, transform=t, limit = -1)
    inf_model.eval()
    if device == "cuda":
        inf_model.cuda()
    with torch.no_grad():
        data_matrix = torch.Tensor().cuda()
        bs = 128
        if len(dataset) < bs:
            bs = 1
        loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        for i, batch in enumerate(loader):
            x = batch[0].cuda() if device == "cuda" else batch[0]
            embeddings = inf_model(x)
            data_matrix = torch.cat([data_matrix, embeddings])

    return data_matrix.cpu().detach().numpy()


def index_gen(embeddings):
    d = embeddings.shape[-1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, "index.bin")
    print("Index created. Stored as index.bin")
    return index

def get_embedding(model, im):
    def to_tensor(pil):
        return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

    t = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.Lambda(to_tensor)]
    )
    model.eval()
    if device == "cuda":
        model.cuda()

    im = Image.open(im) if isinstance(im, str) else im
    datapoint = (
        t(im).unsqueeze(0).cuda() if device == "cuda" else t(im).unsqueeze(0)
    )  # only a single datapoint so we unsqueeze to add a dimension

    with torch.no_grad():
        embedding = model(datapoint)  # get_embedding

    return embedding.detach().cpu().numpy()
