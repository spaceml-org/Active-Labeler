import shutil
import torch
import requests
from IPython import get_ipython
from IPython import get_ipython
from tqdm.notebook import tqdm
from torchvision import transforms
import os
import pathlib
from imutils import paths
import json
import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.image as mpimg
from imutils import paths
import time
import yaml
import random
import torchvision
from torchvision import transforms
from sys import exit

##sim search
from torch.utils.data import DataLoader
from annoy import AnnoyIndex
from torchvision import transforms
from argparse import ArgumentParser

import torchvision.datasets as datasets
import torch
import pickle
import os
from tqdm.notebook import tqdm
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import pandas as pd

import shutil

import PIL.Image as Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from annoy import AnnoyIndex
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.image as mpimg
from imutils import paths
import sys

import logging
logging.info("APP START")

# sys.path.insert(0, "Self-Supervised-Learner")
# sys.path.insert(0, "./ActiveLabeler-main")
# sys.path.insert(0, "./ActiveLabeler-main/Self-Supervised-Learner")
# sys.path.insert(0, "./ActiveLabeler-main/ActiveLabelerModels")

from models import CLASSIFIER
from models import SIMCLR
from models import SIMSIAM
from ActiveLabeler import ActiveLabeler
from TrainModels import TrainModels

# from SimilaritySearch import SimilaritySearch
# from Diversity_Algorithm.diversity_func import Diversity


class Pipeline:
    def __init__(self, config_path, class_name):
        self.config_path = config_path
        self.dataset_paths = [] #contains image names corresponding to emb
        self.unlabeled_list = []
        self.labled_list = []
        self.embeddings = None
        self.div_embeddings = None
        self.initialize_emb_counter = 0
        self.class_name = class_name
        self.metrics = {
            "class": [],
            "step": [],
            "model_type": [],
            "train_ratio": [],
            "pos_train_img": [],
            "neg_train_imgs": [],
            "train_time": [],
            "labled_pos": [],
            "labled_neg": [],
            "f1_score": [],
            "precision": [],
            "accuracy": [],
            "recall": [],
            "pos_class_confidence_0.8": [],
            "pos_class_confidence_0.5": [],
            "pos_class_confidence_median": [],
            "neg_class_confidence_0.8": [],
            "neg_class_confidence_0.5": [],
            "neg_class_confidence_median": [],
            "total_labeling_effort": [],
            "actual_pos_imgs_0.8": [],
            "actual_pos_imgs_0.5": [],
        }
        self.prediction_prob = {}
        #todo
        self.parameters= {}

    # similiarity search class
    def get_annoy_tree(self, num_nodes, embeddings, num_trees, annoy_path):
        t = AnnoyIndex(num_nodes, "euclidean")
        for i in range(len(embeddings)):
            t.add_item(i, embeddings[i])
        t.build(num_trees)
        t.save(annoy_path)
        print("Annoy file stored at ", annoy_path)

    def inference(self, image_path, model, model_type):
        im = Image.open(image_path).convert("RGB")
        image = np.transpose(im, (2, 0, 1)).copy()
        im = torch.tensor(image).unsqueeze(0).float().cuda()
        x = model(im) if model_type == "model" else model.encoder(im)[-1]
        return x[0]

    def get_nn_annoy(
        self,
        image_path,
        n_closest,
        model,
        emb
    ):
        # load dependencies
        u = AnnoyIndex(self.parameters["model"]["embedding_size"], "euclidean")
        u.load(self.parameters["annoy"]["annoy_path"])

        # if emb not passed use inference function and model to generate emb
        image_embedding = self.inference(image_path, model, "model") if emb is None else emb

        inds, dists = u.get_nns_by_vector(
            image_embedding, n_closest, include_distances=True
        )
        return inds, dists

    def generate_embeddings(
        self, image_size, embedding_size, model, dataset_imgs, model_type="model"
    ):
        dataset_paths = [(self.parameters["data"]["data_path"] + "/Unlabeled/" + image_name) for image_name in dataset_imgs]
        t = transforms.Resize((image_size, image_size))
        embedding_matrix = torch.empty(size=(0, embedding_size)).cuda()
        model = model
        for f in tqdm(dataset_paths):
            with torch.no_grad():
                im = Image.open(f).convert("RGB")
                im = t(im)
                im = np.asarray(im).transpose(2, 0, 1)
                im = torch.Tensor(im).unsqueeze(0).cuda()
                if model_type == "model":
                    embedding = model(im)[0]
                else:  # encoder
                    embedding = model.encoder(im)[-1]
                embedding_matrix = torch.vstack((embedding_matrix, embedding))
        logging.info(f"Got embeddings. Embedding Shape: {embedding_matrix.shape}")
        print(f"\nGot embeddings. Embedding Shape: {embedding_matrix.shape}")
        return embedding_matrix.detach().cpu().numpy()

    def initialize_embeddings(
        self,
        image_size,
        embedding_size,
        model,
        dataset_paths,
        num_nodes,
        num_trees,
        annoy_path,
        model_type="model",
    ):
        #generates emb and indexes annoy tree
        self.embeddings = self.generate_embeddings(
            image_size, embedding_size, model, dataset_paths, model_type
        )
        self.get_annoy_tree(
            embedding_size, self.embeddings, num_trees, annoy_path
        )

    def search_similar(
        self, ref_imgs, n_closest, model,embs
    ):
        image_names = set()
        i=0
        for image_path in ref_imgs:
            inds, dists = self.get_nn_annoy(
                image_path,
                n_closest,
                model,
                embs[i]
            )
            i+=1
            for idx in inds:
                image_names.add(self.dataset_paths[idx])
        return list(image_names)

    def label_data(
        self,
        imgs_to_label
    ):

        logging.info("Deduplicate imgs_to_label and prepare for labeling")
        image_names = [image_path.split("/")[-1] for image_path in imgs_to_label]
        image_names = list(set(image_names))
        #remove from image_names if already labeled
        for labled in self.labled_list:
            if labled in image_names:
                image_names.remove(labled)

        for image_name in image_names:
            #copy image to be labled into unlabeled path
            image_path_copy = (
                self.parameters["data"]["data_path"] + "/Unlabeled/" + image_name
            )
            shutil.copy(image_path_copy, self.parameters["nn"]["unlabled_path"])

            #remove from unlabeled list and add to labeled list
            self.unlabeled_list.remove(
                image_name
            )
            self.labled_list.append(image_name)

        logging.debug(f"images sent to labeling: {image_names}")
        self.swipe_label()

    def swipe_label(self):

        unlabled_path = self.parameters["nn"]["unlabled_path"]
        labeled_path =  self.parameters["nn"]["labeled_path"]
        positive_path = self.parameters["nn"]["positive_path"]
        negative_path = self.parameters["nn"]["negative_path"]
        unsure_path = self.parameters["nn"]["unsure_path"]

        logging.info("Calling swipe labeler")
        print(
            f"\n {len(list(paths.list_images(unlabled_path)))} images to label."
        )

        ori_labled = len(list(paths.list_images(labeled_path)))
        ori_pos = len(list(paths.list_images(positive_path)))
        ori_neg = len(list(paths.list_images(negative_path)))

        #simulate labeling
        if self.parameters["nn"]["simulate_label"]:
            for img in list(paths.list_images(unlabled_path)):
                src = unlabled_path + "/" + img.split("/")[-1]
                dest = (
                    (positive_path + "/" + img.split("/")[-1])
                    if self.class_name in img
                    else (negative_path + "/" + img.split("/")[-1])
                )
                shutil.move(src, dest)

        #swipe labeler
        else:
            batch_size = min(len(list(paths.list_images(unlabled_path))),self.parameters['nn']['swipelabel_batch_size'])
            swipe_dir = os.path.join(self.parameters['nn']['swipe_dir'],'Swipe-Labeler-main/api/api.py')
            label = f"python3 {swipe_dir} --path_for_unlabeled='{unlabled_path}' --path_for_pos_labels='{positive_path}' --path_for_neg_labels='{negative_path}' --path_for_unsure_labels='{unsure_path}' --batch_size={batch_size} > swipelog.txt"
            logging.debug(label)
            ossys = os.system(label)
            logging.debug(f"swipe labeler exit code {ossys}")


            # label = f"nohup python3 {swipe_dir} --path_for_unlabeled='{unlabled_path}' --path_for_pos_labels='{positive_path}' --path_for_neg_labels='{negative_path}' --path_for_unsure_labels='{unsure_path}' --batch_size={batch_size} > swipelog.txt &"
            # #todo swipelog merge to main log
            # # >/dev/null 2>&1"
            # logging.debug(label)
            # ossys = os.system(label)
            # print("swipe labeler exit code", ossys)
            # if self.parameters['colab']:
            #     get_ipython().getoutput('lt --port 5000')
            # else:
            #     os.system('lt --port 5000')



        print(
            f" {len(list(paths.list_images(labeled_path))) - ori_labled} labeled: {len(list(paths.list_images(positive_path))) - ori_pos} Pos {len(list(paths.list_images(negative_path))) - ori_neg} Neg"
        )

        logging.info(
            f"{len(list(paths.list_images(labeled_path)))} labeled: {len(list(paths.list_images(positive_path)))} Pos {len(list(paths.list_images(negative_path)))} Neg"
        )
        logging.info(f"unlabeled list: {self.unlabeled_list}")
        logging.info(f"labeled list: {self.labled_list}")

    def create_seed_dataset(
        self,
        model
    ):
        iteration = 0
        n_closest = 1
        while True:
            iteration += 1
            print(f"\n----- iteration: {iteration}")

            print("Enter n closest")
            n_closest = int(input())
            if n_closest == 0:
                break

            ref_imgs = (
                [self.parameters["nn"]["ref_img_path"]] if iteration == 1 else list(paths.list_images(self.parameters["nn"]["positive_path"]))
            )
            embs = [None] if iteration == 1 else self.find_emb(ref_imgs)
            imgs = self.search_similar(
                ref_imgs,
                (n_closest * 8) // 10,
                model,
                embs
            )

            # random sampling 80:20
            n_20 = len(imgs) // 4
            tmp = list(set(self.unlabeled_list) - set(imgs))
            r_imgs = random.Random(self.parameters["seed"]).sample(tmp, k=n_20)
            imgs = list(imgs + r_imgs)

            self.label_data(
                imgs,
            )

    def find_emb(self, ref_imgs):
        emb_ind = [self.dataset_paths.index(img_path.split('/')[-1]) for img_path in ref_imgs]
        embs = [self.embeddings[i] for i in emb_ind]
        return embs


    def load_config(self, config_path):
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config

    def load_model(self, model_type, model_path, data_path):  # , device):
        # todo device
        if model_type == "simclr":
            model = SIMCLR.SIMCLR.load_from_checkpoint(model_path, DATA_PATH=data_path)
            logging.info("simclr model loaded")

        elif model_type == "simsiam":
            model = SIMSIAM.SIMSIAM.load_from_checkpoint(
                model_path, DATA_PATH=data_path
            )
            logging.info("simsiam model loaded")

        model.to("cuda")
        model.eval()
        return model

    def create_emb_label_mapping(self, positive_path, negative_path):
        # emb_dataset = [[emb,label]..] 0-neg, 1 -pos
        emb_dataset = []
        if positive_path is None:
            pos_label = []
        else:
            pos_label = [
                i.split("/")[-1] for i in list(paths.list_images(positive_path))
            ]
        if negative_path is None:
            neg_label = []
        else:
            neg_label = [
                i.split("/")[-1] for i in list(paths.list_images(negative_path))
            ]
        i = -1
        for img_path in self.dataset_paths:
            i = i + 1
            if img_path.split("/")[-1] in pos_label:
                label = 1
                emb_dataset.append([self.embeddings[i], label])
            if img_path.split("/")[-1] in neg_label:
                label = 0
                emb_dataset.append([self.embeddings[i], label])
        return emb_dataset

    def create_emb_list(self, img_names):
        # creates list of emb corresponding to img_names list
        emb_dataset = []
        for img in img_names:
            emb_dataset.append(self.embeddings[self.dataset_paths.index(img.split("/")[-1])])
        return emb_dataset

    def test_data(self, model, test_path, t, device="cuda"):

        test_dataset = torchvision.datasets.ImageFolder(test_path, t)
        loader = DataLoader(test_dataset, batch_size=1)

        # model.to(device)
        model.eval()
        op = []
        gt = []
        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                feats = model.encoder(x)[-1]
                feats = feats.view(feats.size(0), -1)
                output = model.linear_model(feats)
                inds = torch.argmax(output, dim=1)
                # op.append(inds.item())
                op.append(output.detach().cpu().numpy())
                gt.append(y.item())
            pred = []
            for i in op:
                if i[0] <= 0.5:
                    pred.append(0)
                else:
                    pred.append(1)
            op = pred
            prec = precision_score(gt, op)
            rec = recall_score(gt, op)
            f1 = f1_score(gt, op)
            acc = accuracy_score(gt, op)

            self.metrics["f1_score"].append(f1)
            self.metrics["precision"].append(prec)
            self.metrics["recall"].append(rec)
            self.metrics["accuracy"].append(acc)

    @property
    def main(self):
        # offline
        # TODO printing and logging

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        # seed dataset

        logging.info("load config")
        parameters = self.load_config(self.config_path)
        #todo
        self.parameters = self.load_config(self.config_path)

        logging.info("load model")
        model = self.load_model(
            parameters["model"]["model_type"],
            parameters["model"]["model_path"],
            parameters["data"]["data_path"],
        )

        tmp = list(paths.list_images(parameters["data"]["data_path"]))
        self.unlabeled_list = [i.split("/")[-1] for i in tmp]
        self.dataset_paths = [i.split("/")[-1] for i in tmp]

        logging.info("initialize_embeddings")
        self.initialize_embeddings(
            parameters["model"]["image_size"],
            parameters["model"]["embedding_size"],
            model,
            self.dataset_paths,
            parameters["model"]["embedding_size"],
            parameters["annoy"]["num_trees"],
            parameters["annoy"]["annoy_path"],
        )


        #todo continuation
        if parameters["seed_dataset"]["nn"] == 1:
            logging.info("create_seed_dataset")
            self.labled_list = []
            self.create_seed_dataset(
                model,
            )
            newly_labled_path = parameters["nn"]["labeled_path"]

        else:
            self.labled_list = [
                i.split("/")[-1]
                for i in list(
                    paths.list_images(parameters["seed_dataset"]["seed_data_path"])
                )
            ]
            for i in self.labled_list:
                self.unlabeled_list.remove(i)
            newly_labled_path = parameters["seed_dataset"]["seed_data_path"] #todo: put seed inside labeled path ?

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        # AL - linear and finetuning

        os.chdir(parameters["AL_main"]["al_folder"])
        logging.info("active_labeling")
        logging.info(
            "Initializing active labeler and train models class objects."
        )
        #note: whatever unlabled images left has to be updated when not using diversity and using entire dataset
        #pass emb mapping, unlabled images paths
        activelabeler = ActiveLabeler(
            self.create_emb_list(self.unlabeled_list),
            [
                parameters["data"]["data_path"] + "/Unlabeled/" + image_name
                for image_name in self.unlabeled_list
            ],
            self.parameters['model']['image_size'],
            self.parameters['ActiveLabeler']['batch_size'],
            self.parameters['seed']
        )

        train_models = TrainModels(
            parameters["TrainModels"]["config_path"],
            "./", #todo check if this is right or put in config file
            parameters["data"]["data_path"],
            "AL",
        )  # todo saved model path # datapath => sub directory structure for datapath arg

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose(
            [
                transforms.Resize(
                    (
                        parameters["model"]["image_size"],
                        parameters["model"]["image_size"],
                    )
                ),
                transforms.Lambda(to_tensor),
            ]
        )


        iteration = 0
        model_type = "model"
        while True:
            iteration += 1
            print(f"iteration {iteration}")

            print(
                "Enter l for Linear, f for finetuning and q to quit"
            )

            input_counter = input()

            #todo automatic iteration
            # input_counter = "f"
            # if iteration == 21:
            #     input_counter = "q"

            if input_counter == "q":
                break

            if input_counter == "l":
                # train linear = create dataloader on (newly labled + archive dataset) & split into training plus validation, train

                #create newly labeled images emb mapping
                emb_dataset = self.create_emb_label_mapping(
                    newly_labled_path + "/positive/", newly_labled_path + "/negative/"
                )

                #create archive labeled images emb mapping
                if iteration == 1:
                    emb_dataset_archive = self.create_emb_label_mapping(
                        parameters["nn"]["labeled_path"] + "/positive",
                        parameters["nn"]["labeled_path"] + "/negative",
                    )

                else:
                    emb_dataset_archive = self.create_emb_label_mapping(
                        parameters["AL_main"]["archive_path"] + "/positive",
                        parameters["AL_main"]["archive_path"] + "/negative",
                    )

                #newly labled + archive emb mapping
                for i in emb_dataset_archive:
                    emb_dataset.append(i)

                #add train ratio metrics
                if self.parameters['test']['metrics']:
                    tmp_p = len(
                        list(paths.list_images(newly_labled_path + "/positive"))
                    ) + len(
                        list(
                            paths.list_images(
                                parameters["AL_main"]["archive_path"] + "/positive"
                            )
                        )
                    )
                    tmp_n = len(
                        list(paths.list_images(newly_labled_path + "/negative"))
                    ) + len(
                        list(
                            paths.list_images(
                                parameters["AL_main"]["archive_path"] + "/negative"
                            )
                        )
                    )
                    self.metrics["pos_train_img"].append(tmp_p)
                    self.metrics["neg_train_imgs"].append(tmp_n)
                    tmp = tmp_n / tmp_p if tmp_p > 0 else 0
                    self.metrics["train_ratio"].append(tmp)

                #training and validation dataset
                emb_dataset = random.Random(self.parameters["seed"]).sample(emb_dataset, len(emb_dataset))
                n_80 = (len(emb_dataset) * 8) // 10
                training_dataset = DataLoader(
                    emb_dataset[:n_80], batch_size=self.parameters['AL_main']['train_dataset_batch_size']
                )
                validation_dataset = DataLoader(emb_dataset[n_80 + 1 :], batch_size=1)

                #train linear
                tic = time.perf_counter()
                train_models.train_linear(training_dataset, validation_dataset)
                toc = time.perf_counter()
                if self.parameters['test']['metrics']:
                    self.metrics["train_time"].append((toc - tic) // 60)

            # put seed dataset/newly labeled data in archive path and clear newly labeled
            for img in list(paths.list_images(newly_labled_path + "/positive")):
                shutil.copy(img, parameters["AL_main"]["archive_path"] + "/positive")
            for img in list(paths.list_images(newly_labled_path + "/negative")):
                shutil.copy(img, parameters["AL_main"]["archive_path"] + "/negative")
            newly_labled_path = parameters["AL_main"]["newly_labled_path"]
            for img in list(paths.list_images(newly_labled_path)):
                os.remove(img)

            if input_counter == "f":
                model_type = "encoder"
                # train all = create dataloader on archive dataset & split into training plus validation, train all, regenerate emb

                #add train ratio metrics
                if self.parameters['test']['metrics']:
                    tmp_p = len(
                        list(
                            paths.list_images(
                                parameters["AL_main"]["archive_path"] + "/positive"
                            )
                        )
                    ) + len(list(paths.list_images(newly_labled_path + "/positive")))
                    tmp_n = len(
                        list(
                            paths.list_images(
                                parameters["AL_main"]["archive_path"] + "/negative"
                            )
                        )
                    ) + len(list(paths.list_images(newly_labled_path + "/negative")))
                    self.metrics["pos_train_img"].append(tmp_p)
                    self.metrics["neg_train_imgs"].append(tmp_n)
                    tmp = tmp_n / tmp_p if tmp_p > 0 else 0
                    self.metrics["train_ratio"].append(tmp)

                #training and validation datasets
                archive_dataset = torchvision.datasets.ImageFolder(
                    parameters["AL_main"]["archive_path"], t
                )
                n_80 = (len(archive_dataset) * 8) // 10
                n_20 = len(archive_dataset) - n_80
                training_dataset, validation_dataset = torch.utils.data.random_split(
                    archive_dataset, [n_80, n_20]
                )
                training_dataset = DataLoader(training_dataset, batch_size=self.parameters['AL_main']['train_dataset_batch_size'])
                validation_dataset = DataLoader(validation_dataset, batch_size=1)

                tic = time.perf_counter()
                train_models.train_all(training_dataset, validation_dataset)
                toc = time.perf_counter()
                if self.parameters['test']['metrics']:
                    self.metrics["train_time"].append((toc - tic) // 60)

                # todo ? change generate emb again => using encoder from model from train_all
                logging.info("regenerate embeddings")
                encoder = train_models.get_model().to("cuda")  # todo device
                self.initialize_embeddings(
                    parameters["model"]["image_size"],
                    parameters["model"]["embedding_size"],
                    encoder,
                    self.dataset_paths,
                    parameters["model"]["embedding_size"],
                    parameters["annoy"]["num_trees"],
                    parameters["annoy"]["annoy_path"],
                    "encoder",
                )

                # update AL class with new unlabled emb mapping
                mapping = self.create_emb_list(self.unlabeled_list)
                activelabeler.get_embeddings_offline(
                    mapping,
                    [
                        parameters["data"]["data_path"] + "/Unlabeled/" + image_name
                        for image_name in self.unlabeled_list
                    ],
                )

            if input_counter != "f" and input_counter != "l":
                continue

            # AL.getimgstolabel => uncertain imgs => nn sampling_strategy
            curr_model = model if model_type == "model" else encoder #the latest encoder is used

            (
                strategy_embeddings,
                strategy_images,
            ) = activelabeler.get_images_to_label_offline(
                train_models.get_model(),
                parameters["ActiveLabeler"]["sampling_strategy"],
                parameters["ActiveLabeler"]["sample_size"],
                None,
                "cuda"
            )

            #nn and label
            if parameters["AL_main"]["nn"] == 1:
                embs = self.find_emb(strategy_images)
                imgs = self.search_similar(
                    strategy_images,
                    int(parameters["AL_main"]["n_closest"]),
                    curr_model,
                    embs,
                )
                print("nn imgs ", imgs)
                tmp2 = set(imgs)
                print("len nn", len(tmp2))
                tmp2.update(strategy_images)
                imgs = list(tmp2)
                print("len nn + strategy imgs", len(tmp2))

            else:
                imgs = strategy_images

            self.parameters['nn']['labeled_path'] =parameters['AL_main']['newly_labled_path']
            self.parameters['nn']['positive_path'] = parameters['AL_main']['newly_labled_path'] + "/positive"
            self.parameters['nn']['negative_path']= parameters['AL_main']['newly_labled_path'] + "/negative"

            self.label_data(imgs)

            #image metrics
            tmp1 = len(
                list(
                    paths.list_images(
                        parameters["AL_main"]["archive_path"] + "/positive"
                    )
                )
            )
            tmp2 = len(list(paths.list_images(newly_labled_path + "/positive")))
            tmp3 = len(
                list(
                    paths.list_images(
                        parameters["AL_main"]["archive_path"] + "/negative"
                    )
                )
            )
            tmp4 = len(list(paths.list_images(newly_labled_path + "/negative")))
            print(
                f"Total Images: {tmp1} + {tmp2} = {tmp1+tmp2} positive || {tmp3} + {tmp4} = {tmp3+tmp4} negative"
            )
            if self.parameters['test']['metrics']:
                self.metrics["labled_pos"].append(tmp2)
                self.metrics["labled_neg"].append(tmp4)


            #get model predictions and corresponding imgs
            predic_prob, predic_prob_imgs = activelabeler.get_prob()


            #updating AL class with latest unlabled images and emb - done for both l and f
            mapping = self.create_emb_list(self.unlabeled_list)
            activelabeler.get_embeddings_offline(
                mapping,
                [
                    parameters["data"]["data_path"] + "/Unlabeled/" + image_name
                    for image_name in self.unlabeled_list
                ],
            )

            # --TEST
            if parameters['test']['metrics']:

                # step, class, model_type append in main
                self.metrics["step"].append(iteration)
                self.metrics["class"].append(self.class_name)
                self.metrics["model_type"].append(input_counter)

                self.test_data(train_models.get_model(), parameters["test"]["test_path"], t)

                #prob metrics

                # find label for each prediction
                prob_pos, prob_neg = [], []
                i = 0
                for p in predic_prob:
                    if self.class_name in predic_prob_imgs[i]:
                        prob_pos.append(p)
                    else:
                        prob_neg.append(p)
                    i += 1

                count_8 = 0
                count_5 = 0
                for p in prob_pos:
                    if p >= 0.8:
                        count_8 += 1
                    if p >= 0.5:
                        count_5 += 1
                self.metrics["pos_class_confidence_0.8"].append(count_8)
                self.metrics["pos_class_confidence_0.5"].append(count_5)
                self.metrics["pos_class_confidence_median"].append(np.median(prob_pos))

                count_8 = 0
                count_5 = 0
                for p in prob_neg:
                    if p >= 0.8:
                        count_8 += 1
                    if p >= 0.5:
                        count_5 += 1
                self.metrics["neg_class_confidence_0.8"].append(count_8)
                self.metrics["neg_class_confidence_0.5"].append(count_5)
                self.metrics["neg_class_confidence_median"].append(np.median(prob_neg))

                self.metrics["actual_pos_imgs_0.8"].append(
                    self.metrics["pos_train_img"][-1]
                    + self.metrics["pos_class_confidence_0.8"][-1]
                    + self.metrics["labled_pos"][-1]
                )
                self.metrics["actual_pos_imgs_0.5"].append(
                    self.metrics["pos_train_img"][-1]
                    + self.metrics["pos_class_confidence_0.5"][-1]
                    + self.metrics["labled_pos"][-1]
                )
                self.metrics["total_labeling_effort"].append(
                    self.metrics["pos_train_img"][-1]
                    + self.metrics["neg_train_imgs"][-1]
                    + self.metrics["pos_class_confidence_0.8"][-1]
                    + self.metrics["neg_class_confidence_0.8"][-1]
                )

                prob_pos.extend(prob_neg)
                self.prediction_prob[iteration] = prob_pos
                df = pd.DataFrame.from_dict(
                    self.prediction_prob, orient="index"
                ).transpose()
                df.to_csv(parameters["test"]["prob_csv_path"], index=False)

                print(f"iteration {iteration} metrics = {self.metrics}")
                df = pd.DataFrame.from_dict(self.metrics, orient="index").transpose()
                # rounding to 2
                col_names = [
                    "f1_score",
                    "precision",
                    "accuracy",
                    "recall",
                    "train_ratio",
                    "pos_class_confidence_median",
                    "neg_class_confidence_median",
                ]
                for i in col_names:
                    df[i] = df[i].astype(float).round(2)
                df.to_csv(parameters["test"]["metric_csv_path"], index=False)