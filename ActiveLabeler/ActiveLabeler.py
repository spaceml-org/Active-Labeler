import os
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

class ActiveLabeler():
    def __init__(self, embeddings, images_path):
        self.embeddings = embeddings
        self.images_path = images_path
    
    def strategy(self, name, query, N, strategy_params):
        """Method used to get a subset of unlabled images.

        :param name: Name of the sampling strategy to use
        :options name: uncertainty, random, positive

        :param query: A num_images X num_classes sized numpy array. This numpy array has the softmax probability 
                    of each image. These softmax probs are predicted by the latest finetune model.
        :type query: numpy array
        
        :param N: Number of images that should be in the subset.
        :type N: integer

        :returns list/numpy array of the index of images in the sebset. 
        """
        if name == 'uncertainty':
            return np.argsort(query.std(axis = 1))[:N]
        elif name == 'random':
            return [random.randrange(0, len(query), 1) for i in range(N)]
        elif name == 'positive':
            positive_predictions = np.array([query[i][strategy_params["class_label"]] for i in range(len(query))])
            positive_index = np.argsort(positive_predictions)
            return positive_index[::-1][:N]
        else:
            raise NotImplementedError

    def get_embeddings_offline(self, embeddings_path):
        #self.embeddings Lists
        #self.images_path Lists
        pass #TODO

    def get_images_to_label_offline(self, model, sampling_strat, sample_size, strategy_params, device):
        #Load stuff
        model.eval()
        if device == "cuda":
            model.cuda()
        dataset = self.embeddings
        image_paths = self.images_path

        #Forward Pass
        with torch.no_grad():
            bs = 128
            if len(dataset) < bs:
                bs = 1
            loader = DataLoader(dataset, batch_size=bs, shuffle=False)
            model_predictions = []
            for batch in tqdm(loader):
                if device == "cuda":
                    x = torch.cuda.FloatTensor(batch)
                else:
                    x = torch.FloatTensor(batch)
                predictions = model(x)
                model_predictions.extend(predictions) 
        
        #Strategy
        model_predictions = np.array(model_predictions)
        subset = self.strategy(sampling_strat, model_predictions, sample_size, strategy_params)

        #Stuff to return
        strategy_embeddings = np.array([i for i in dataset])[subset]
        strategy_images = np.array([i for i in image_paths])[subset]

        return strategy_embeddings, strategy_images