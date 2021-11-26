import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ActiveLabeler:
    """Active labeler code which is used with a SSL model to get curated dataset from unlabeled data"""

    def __init__(self, embeddings, images_path,image_size,batch_size,seed):
        """Initialise with embeddings and respected image paths"""
        self.embeddings = embeddings
        self.images_path = images_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.probabilities = []
        np.random.seed(seed)
        random.seed(seed)

    def strategy(self, name, query, N, strategy_params):
        """
        Get a subset of unlabled images picked by a specific strategy

        Keyword arguments
        name -- Name of the sampling strategy to use
                Options: uncertainty, uncertainty_balanced, random, positive, gaussian

        query -- A num_images X num_classes sized numpy array. This numpy array has the softmax probability
                    of each image. These softmax probs are predicted by the latest finetune model. Type: numpy array

        N -- Number of images that should be in the subset. Type: integer

        strategy_params -- A dictionary of strategy specific args. Type: Dictionary

        Returns list/numpy array of the index of images in the subset.
        """
        if name == "gaussian":

            def gaussian(x, mean, sigma):
                a = 1 / (sigma * np.sqrt(2 * np.pi))
                exp_val = -(1 / 2) * (((x - mean) ** 2) / (sigma ** 2))
                tmp = list(a * np.exp(exp_val))[0]
                return tmp

            def get_gaussian_weights(confidences):
                mean = 0.5
                sigma = 0.2
                return [gaussian(conf_i, mean, sigma) for conf_i in confidences]

            def get_samples(image_list, confidences, number_of_samples):
                sample_wts = get_gaussian_weights(confidences)
                sample_wts_norm = np.array(sample_wts)
                sample_wts_norm /= sample_wts_norm.sum()
                picked_samples = np.random.choice(
                    image_list, replace=False, size=number_of_samples, p=sample_wts_norm
                )
                return picked_samples

            def get_index(samples, dataset):
                data_index = []
                for i in samples:
                    data_index.append(dataset.index(i))
                data_index = np.array(data_index)
                return dataset[data_index]

            selected_samples = get_samples(self.images_path, query, N)
            # selected_embeddinds = get_index(selected_samples, self.embeddings)
            return selected_samples
        elif name == "uncertainty":
            tmp_query = np.array([query[i][0] for i in range(len(query))])
            difference_array = np.absolute(tmp_query - 0.5)
            return difference_array.argsort()[:N]
        elif name == "uncertainty_balanced":
            tmp_query = np.array([query[i][0] for i in range(len(query))])
            difference_array = tmp_query - 0.5
            sorted_diff = difference_array.argsort()
            idx_0 = np.absolute(difference_array).argsort()[0]
            idx_sorted_0 = list(sorted_diff).index(idx_0)

            #todo logs
            print("tmp_query- probs - sorted", sorted(tmp_query))
            print("diff arr ", difference_array)
            print("diff arr sorted ", np.sort(difference_array))
            print("middle value idx", idx_sorted_0)
            print("middle value ", difference_array[idx_sorted_0])

            N = N - 1
            # take n/2 from either side of index_sorted_0, less ele
            print("splitting idices", (idx_sorted_0 - (N // 2) - 1), idx_sorted_0)
            print("splitting idices", idx_sorted_0 + 1, (idx_sorted_0 + (N // 2)))
            tmp1 = sorted_diff[(idx_sorted_0 - (N // 2)) : idx_sorted_0]
            tmp2 = sorted_diff[idx_sorted_0 + 1 : (idx_sorted_0 + (N // 2)) + 1]
            print("tmp1", tmp1)
            print("tmp2", tmp2)
            print(
                f"uncertainity balanced: {len(tmp1)} + {len(tmp2)} = {len(tmp1) + len(tmp2)}"
            )
            # returning the corresponding indexes tmp1 + tmp2
            tmp3 = []
            for i in tmp1:
                tmp3.append(i)
            for i in tmp2:
                tmp3.append(i)
            tmp3.append((idx_0))
            return tmp3
        elif name == "random":
            return [random.randrange(0, len(query), 1) for i in range(N)]
        elif name == "positive":
            positive_predictions = np.array(
                [query[i][strategy_params["class_label"]] for i in range(len(query))]
            )
            positive_index = np.argsort(positive_predictions)
            return positive_index[::-1][:N]
        else:
            raise NotImplementedError

    def get_embeddings_offline(self, emb, data_paths):
        """
        Load the embeddings and the related image paths.

        Keyword arguments
        emb -- A numpy array of data embeddings
        data_paths -- A list of related image paths
        """
        self.embeddings = emb
        self.images_path = data_paths



    def get_images_to_label_offline(
        self, model, sampling_strat, sample_size, strategy_params, device
    ):
        """
        Use a classification model to pick new images to label for model finetuning accourding to a strategy function

        Keyword arguments
        model -- A pytorch classification model
        sampling_strat -- Strategy name
        sample_size -- Number of data points that strategy function should pick
        strategy_params -- Additional strategy parameters
        device -- cudo (GPU) or cpu
        data_type -- Images or embeddings
        """
        model.eval()
        if device == "cuda":
            model.cuda()

        dataset = self.embeddings
        image_paths = self.images_path
        with torch.no_grad():
            if len(dataset) < self.batch_size:
                self.batch_size = 1
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            model_predictions = []
            for batch in tqdm(loader):
                if device == "cuda":
                    x = torch.FloatTensor(batch).to(device)
                else:
                    x = torch.FloatTensor(batch)
                x = x.view(x.size(0), -1)
                predictions = model.linear_model(x)
                model_predictions.extend(predictions.cpu().detach().numpy())
            model_predictions = np.array(model_predictions)
            self.probabilities = model_predictions
            subset = self.strategy(
                sampling_strat, model_predictions, sample_size, strategy_params
            )
            if sampling_strat == "gaussian":
                strategy_embeddings = []
                strategy_images = subset
            else:
                strategy_embeddings = np.array([i for i in dataset])[subset]
                strategy_images = np.array([i for i in image_paths])[subset]
            return strategy_embeddings, strategy_images

    def get_prob(self):
        """Returns the model predictions"""
        return self.probabilities, self.images_path

