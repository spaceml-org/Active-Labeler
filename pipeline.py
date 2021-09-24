import sys
sys.path.insert(0, "./Active-Labeler")
sys.path.insert(0, "./Active-Labeler/ActiveLabeler-main")
sys.path.insert(0, "./Active-Labeler/ActiveLabeler-main/Self-Supervised-Learner")
sys.path.insert(0, "./Active-Labeler/ActiveLabeler-main/ActiveLabelerModels")

import pathlib
import yaml
import random
import torchvision
import os
from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
import pandas as pd
import shutil
import PIL.Image as Image
import torch
import numpy as np
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
import logging

from models import SIMCLR
from models import SIMSIAM
from ActiveLabeler import ActiveLabeler
from TrainModels import TrainModels

class Pipeline:
    def __init__(self, config_path):
        print("Initialization")
        self.config_path = config_path
        self.dataset_paths = [] #contains image names corresponding to emb
        self.unlabeled_list = []
        self.labled_list = []
        self.embeddings = None
        self.div_embeddings = None
        self.initialize_emb_counter = 0
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

        print("Load Config")
        self.parameters= self.load_config(self.config_path)

        #set seed
        # set seed
        random.seed(self.parameters["seed"])
        np.random.seed(self.parameters["seed"])

        # log settings
        #log_file = os.path.join(self.parameters["runtime_path"],"active_labeler.log")
        log_file = "active_labeler.log"
        if self.parameters["verbose"] == 0:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)-8s - %(funcName)-15s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    logging.FileHandler(log_file, mode='a'),
                ]
            )
        else:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)-8s - %(funcName)-15s - %(message)s',
                datefmt='%d-%b-%y %H:%M:%S',
                handlers=[
                    logging.FileHandler(log_file, mode='a'),
                    logging.StreamHandler()
                ]
            )



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
        if self.parameters["device"] == "cuda":
            im = torch.tensor(image).unsqueeze(0).float().cuda()
        elif self.parameters["device"] == "cpu":
            im = torch.tensor(image).unsqueeze(0).float().cpu()
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
        dataset_paths = [(self.parameters["data_path"] + "/Unlabeled/" + image_name) for image_name in dataset_imgs]
        t = transforms.Resize((image_size, image_size))
        if self.parameters["device"] == "cuda":
            embedding_matrix = torch.empty(size=(0, embedding_size)).cuda()
        elif self.parameters["device"] == "cpu":
            embedding_matrix = torch.empty(size=(0, embedding_size)).cpu()
        model = model
        for f in tqdm(dataset_paths):
            with torch.no_grad():
                im = Image.open(f).convert("RGB")
                im = t(im)
                im = np.asarray(im).transpose(2, 0, 1)
                if self.parameters["device"] == "cuda":
                    im = torch.Tensor(im).unsqueeze(0).cuda()
                elif self.parameters["device"] == "cpu":
                    im = torch.Tensor(im).unsqueeze(0).cpu()
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
                self.parameters["data_path"] + "/Unlabeled/" + image_name
            )
            shutil.copy(image_path_copy, self.parameters["swipe_labeler"]["unlabeled_path"])

            #remove from unlabeled list and add to labeled list
            self.unlabeled_list.remove(
                image_name
            )
            self.labled_list.append(image_name)

        logging.debug(f"images sent to labeling: {image_names}")
        self.swipe_label()

    def swipe_label(self):

        unlabeled_path = self.parameters["swipe_labeler"]["unlabeled_path"]
        labeled_path =  self.parameters["swipe_labeler"]["labeled_path"]
        positive_path = self.parameters["swipe_labeler"]["positive_path"]
        negative_path = self.parameters["swipe_labeler"]["negative_path"]
        unsure_path = self.parameters["swipe_labeler"]["unsure_path"]

        logging.info("Calling swipe labeler")
        print(
            f"\n {len(list(paths.list_images(unlabeled_path)))} images to label."
        )

        ori_labled = len(list(paths.list_images(labeled_path)))
        ori_pos = len(list(paths.list_images(positive_path)))
        ori_neg = len(list(paths.list_images(negative_path)))

        #simulate labeling
        if self.parameters["test"]["simulate_label"]:
            for img in list(paths.list_images(unlabeled_path)):
                src = unlabeled_path + "/" + img.split("/")[-1]
                dest = (
                    (positive_path + "/" + img.split("/")[-1])
                    if self.parameters["test"]["pos_class"] in img
                    else (negative_path + "/" + img.split("/")[-1])
                )
                shutil.move(src, dest)

        #swipe labeler
        else:
            batch_size = min(len(list(paths.list_images(unlabeled_path))),self.parameters['swipe_label_batch_size'])
            swipe_dir = os.path.join("Active-Labeler/",'Swipe-Labeler-main/api/api.py')
            swipe_log = "> swipelabeler.log"
            #swipe_log = f"> {os.path.join(self.parameters['runtime_path'], 'swipelabeler.log')}"
            label = f"python3 {swipe_dir} --path_for_unlabeled='{unlabeled_path}' --path_for_pos_labels='{positive_path}' --path_for_neg_labels='{negative_path}' --path_for_unsure_labels='{unsure_path}' --batch_size={batch_size} {swipe_log}"
            logging.debug(label)
            ossys = os.system(label)
            logging.debug(f"swipe labeler exit code {ossys}")


            # label = f"nohup python3 {swipe_dir} --path_for_unlabeled='{unlabeled_path}' --path_for_pos_labels='{positive_path}' --path_for_neg_labels='{negative_path}' --path_for_unsure_labels='{unsure_path}' --batch_size={batch_size} > swipelog.txt &"
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

            print("Enter n closest, 0 to stop")
            n_closest = int(input())
            if n_closest == 0:
                break

            ref_imgs = (
                [self.parameters["seed_dataset"]["ref_img_path"]] if iteration == 1 else list(paths.list_images(self.parameters["swipe_labeler"]["positive_path"]))
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
        if model_type == "simclr":
            model = SIMCLR.SIMCLR.load_from_checkpoint(model_path, DATA_PATH=data_path)
            logging.info("simclr model loaded")

        elif model_type == "simsiam":
            model = SIMSIAM.SIMSIAM.load_from_checkpoint(
                model_path, DATA_PATH=data_path
            )
            logging.info("simsiam model loaded")

        model.to(self.parameters["device"])
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

    def test_data(self, model, loader, device="cuda"):
        #def test_data(self, model, test_path, t, device="cuda"):

        # test_dataset = torchvision.datasets.ImageFolder(test_path, t)
        # loader = DataLoader(test_dataset, batch_size=1)

        # model.to(device)
        model.eval()
        op = []
        gt = []
        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                output = model(x)
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

    def main(self):
        # offline
        # TODO printing and logging

        # runtime folder sub directories - contains all runtime content
        self.parameters["swipe_labeler"]={}
        self.parameters["swipe_labeler"]["labeled_path"] = os.path.join(self.parameters["runtime_path"],
                                                                        "swipe/labeled")
        self.parameters["swipe_labeler"]["positive_path"] = os.path.join(
            self.parameters["swipe_labeler"]["labeled_path"], "positive")
        self.parameters["swipe_labeler"]["negative_path"] = os.path.join(
            self.parameters["swipe_labeler"]["labeled_path"], "negative")
        self.parameters["swipe_labeler"]["unlabeled_path"] = os.path.join(self.parameters["runtime_path"],
                                                                          "swipe/unlabeled")

        self.parameters["swipe_labeler"]["unsure_path"] = os.path.join(self.parameters["runtime_path"],
                                                                       "swipe/unsure")
        self.parameters["annoy"]["annoy_path"] = os.path.join(self.parameters["runtime_path"],
                                                              "annoy_file.ann")

        self.parameters["ActiveLabeler"]["newly_labled_path"] = os.path.join(self.parameters["runtime_path"],
                                                                             "swipe/new_label")
        self.parameters["ActiveLabeler"]["archive_path"] = os.path.join(self.parameters["runtime_path"],
                                                                        "swipe/archive")
        self.parameters["ActiveLabeler"]["final_dataset_path"] = os.path.join("./final_dataset")

        # creating the directories
        if os.path.exists(self.parameters["runtime_path"]):
            shutil.rmtree(self.parameters["runtime_path"])

        if os.path.exists(self.parameters["ActiveLabeler"]["final_dataset_path"]):
            shutil.rmtree(self.parameters["ActiveLabeler"]["final_dataset_path"])

        for i in [self.parameters["swipe_labeler"]["unlabeled_path"],
                  self.parameters["swipe_labeler"]["positive_path"],
                  self.parameters["swipe_labeler"]["negative_path"], self.parameters["swipe_labeler"]["unsure_path"],
                  os.path.join(self.parameters["ActiveLabeler"]["newly_labled_path"], "positive"),
                  os.path.join(self.parameters["ActiveLabeler"]["newly_labled_path"], "negative"),
                  os.path.join(self.parameters["ActiveLabeler"]["archive_path"], "positive"),
                  os.path.join(self.parameters["ActiveLabeler"]["archive_path"], "negative"),
                  os.path.join(self.parameters["ActiveLabeler"]["final_dataset_path"], "positive"),
                  os.path.join(self.parameters["ActiveLabeler"]["final_dataset_path"], "negative"),]:
            pathlib.Path(i).mkdir(parents=True, exist_ok=True)

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        # seed dataset



        logging.info("load model")
        model = self.load_model(
            self.parameters["model"]["model_type"],
            self.parameters["model"]["model_path"],
            self.parameters["data_path"],
        )

        tmp = list(paths.list_images(self.parameters["data_path"]))
        self.unlabeled_list = [i.split("/")[-1] for i in tmp]
        self.dataset_paths = [i.split("/")[-1] for i in tmp]

        logging.info("initialize_embeddings")
        self.initialize_embeddings(
            self.parameters["model"]["image_size"],
            self.parameters["model"]["embedding_size"],
            model,
            self.dataset_paths,
            self.parameters["model"]["embedding_size"],
            self.parameters["annoy"]["num_trees"],
            self.parameters["annoy"]["annoy_path"],
        )

        if self.parameters["seed_dataset"]["seed_nn"] == 1:
            logging.info("create_seed_dataset")
            self.labled_list = []
            self.create_seed_dataset(
                model,
            )
            newly_labled_path = self.parameters["swipe_labeler"]["labeled_path"]

        else:
            self.labled_list = [
                i.split("/")[-1]
                for i in list(
                    paths.list_images(self.parameters["seed_dataset"]["seed_data_path"])
                )
            ]
            for i in self.labled_list:
                self.unlabeled_list.remove(i)
            newly_labled_path = self.parameters["seed_dataset"]["seed_data_path"]

        # +++++++++++++++++++++++++++++++++++++++++++++++++++++
        # AL - linear and finetuning

        logging.info("active_labeling")
        logging.info(
            "Initializing active labeler and train models class objects."
        )
        #note: whatever unlabled images left has to be updated when not using diversity and using entire dataset
        #pass emb mapping, unlabled images paths
        activelabeler = ActiveLabeler(
            self.create_emb_list(self.unlabeled_list),
            [
                self.parameters["data_path"] + "/Unlabeled/" + image_name
                for image_name in self.unlabeled_list
            ],
            self.parameters['model']['image_size'],
            self.parameters['ActiveLabeler']['active_label_batch_size'],
            self.parameters['seed']
        )

        train_models = TrainModels(
            self.parameters["model"]["model_config_path"],
            "./final_model.ckpt",
            self.parameters["data_path"],
        )

        def to_tensor(pil):
            return torch.tensor(np.array(pil)).permute(2, 0, 1).float()

        t = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self.parameters["model"]["image_size"],
                        self.parameters["model"]["image_size"],
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
                        self.parameters["swipe_labeler"]["labeled_path"] + "/positive",
                        self.parameters["swipe_labeler"]["labeled_path"] + "/negative",
                    )

                else:
                    emb_dataset_archive = self.create_emb_label_mapping(
                        self.parameters["ActiveLabeler"]["archive_path"] + "/positive",
                        self.parameters["ActiveLabeler"]["archive_path"] + "/negative",
                    )

                #newly labled + archive emb mapping  # emb_dataset = [[emb,label]..] 0-neg, 1 -pos
                for i in emb_dataset_archive:
                    emb_dataset.append(i)

                #add train ratio metrics
                if self.parameters['test']['metrics']:
                    tmp_p = len(
                        list(paths.list_images(newly_labled_path + "/positive"))
                    ) + len(
                        list(
                            paths.list_images(
                                self.parameters["ActiveLabeler"]["archive_path"] + "/positive"
                            )
                        )
                    )
                    tmp_n = len(
                        list(paths.list_images(newly_labled_path + "/negative"))
                    ) + len(
                        list(
                            paths.list_images(
                                self.parameters["ActiveLabeler"]["archive_path"] + "/negative"
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
                    emb_dataset[:n_80], batch_size=self.parameters['ActiveLabeler']['train_dataset_batch_size']
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
                shutil.copy(img, self.parameters["ActiveLabeler"]["archive_path"] + "/positive")
            for img in list(paths.list_images(newly_labled_path + "/negative")):
                shutil.copy(img, self.parameters["ActiveLabeler"]["archive_path"] + "/negative")
            newly_labled_path = self.parameters["ActiveLabeler"]["newly_labled_path"]
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
                                self.parameters["ActiveLabeler"]["archive_path"] + "/positive"
                            )
                        )
                    ) + len(list(paths.list_images(newly_labled_path + "/positive")))
                    tmp_n = len(
                        list(
                            paths.list_images(
                                self.parameters["ActiveLabeler"]["archive_path"] + "/negative"
                            )
                        )
                    ) + len(list(paths.list_images(newly_labled_path + "/negative")))
                    self.metrics["pos_train_img"].append(tmp_p)
                    self.metrics["neg_train_imgs"].append(tmp_n)
                    tmp = tmp_n / tmp_p if tmp_p > 0 else 0
                    self.metrics["train_ratio"].append(tmp)

                #training and validation datasets
                archive_dataset = torchvision.datasets.ImageFolder(
                    self.parameters["ActiveLabeler"]["archive_path"], t
                )
                n_80 = (len(archive_dataset) * 8) // 10
                n_20 = len(archive_dataset) - n_80
                training_dataset, validation_dataset = torch.utils.data.random_split(
                    archive_dataset, [n_80, n_20]
                )
                training_dataset = DataLoader(training_dataset, batch_size=self.parameters['ActiveLabeler']['train_dataset_batch_size'])
                validation_dataset = DataLoader(validation_dataset, batch_size=1)

                tic = time.perf_counter()
                train_models.train_all(training_dataset, validation_dataset)
                toc = time.perf_counter()
                if self.parameters['test']['metrics']:
                    self.metrics["train_time"].append((toc - tic) // 60)

                logging.info("regenerate embeddings")
                encoder = train_models.get_model().to(self.parameters["device"])
                self.initialize_embeddings(
                    self.parameters["model"]["image_size"],
                    self.parameters["model"]["embedding_size"],
                    encoder,
                    self.dataset_paths,
                    self.parameters["model"]["embedding_size"],
                    self.parameters["annoy"]["num_trees"],
                    self.parameters["annoy"]["annoy_path"],
                    "encoder",
                )

                # update AL class with new unlabled emb mapping
                mapping = self.create_emb_list(self.unlabeled_list)
                activelabeler.get_embeddings_offline(
                    mapping,
                    [
                        self.parameters["data_path"] + "/Unlabeled/" + image_name
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
                self.parameters["ActiveLabeler"]["sampling_strategy"],
                self.parameters["ActiveLabeler"]["sample_size"],
                None,
                self.parameters["device"]
            )

            #nn and label
            if self.parameters["ActiveLabeler"]["sampling_nn"] == 1:
                embs = self.find_emb(strategy_images)
                imgs = self.search_similar(
                    strategy_images,
                    int(self.parameters["ActiveLabeler"]["n_closest"]),
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

            self.parameters["swipe_labeler"]["labeled_path"] =self.parameters['ActiveLabeler']['newly_labled_path']
            self.parameters["swipe_labeler"]['positive_path'] = self.parameters['ActiveLabeler']['newly_labled_path'] + "/positive"
            self.parameters["swipe_labeler"]['negative_path']= self.parameters['ActiveLabeler']['newly_labled_path'] + "/negative"

            self.label_data(imgs)

            #image metrics
            tmp1 = len(
                list(
                    paths.list_images(
                        self.parameters["ActiveLabeler"]["archive_path"] + "/positive"
                    )
                )
            )
            tmp2 = len(list(paths.list_images(newly_labled_path + "/positive")))
            tmp3 = len(
                list(
                    paths.list_images(
                        self.parameters["ActiveLabeler"]["archive_path"] + "/negative"
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
                    self.parameters["data_path"] + "/Unlabeled/" + image_name
                    for image_name in self.unlabeled_list
                ],
            )

            # --TEST
            if self.parameters['test']['metrics']:

                # step, class, model_type append in main
                self.metrics["step"].append(iteration)
                self.metrics["class"].append(self.parameters["test"]["pos_class"])
                self.metrics["model_type"].append(input_counter)

                #self.test_data(train_models.get_model(), self.parameters["test"]["test_path"], t)
                self.test_data(train_models.get_model(),validation_dataset, self.parameters["device"])

                #prob metrics

                # find label for each prediction
                prob_pos, prob_neg = [], []
                i = 0
                for p in predic_prob:
                    if self.parameters["test"]["pos_class"] in predic_prob_imgs[i]:
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
                df.to_csv(self.parameters["test"]["prob_csv_path"], index=False)

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
                df.to_csv(self.parameters["test"]["metric_csv_path"], index=False)

        # save final model
        train_models.save_model()

        #saving final dataset based on predictions
        unlabeled_predictions, img_paths = activelabeler.get_prob()

        pos_path = os.path.join(self.parameters["ActiveLabeler"]["final_dataset_path"],"positive")
        neg_path = os.path.join(self.parameters["ActiveLabeler"]["final_dataset_path"],"negative")

        for i in range(len(img_paths)):
            if unlabeled_predictions[i] > 0.5:
                target = os.path.join(pos_path, img_paths[i].split("/")[-1])
                shutil.copy(img_paths[i], target)
            else:
                target = os.path.join(neg_path, img_paths[i].split("/")[-1])
                shutil.copy(img_paths[i], target)

        logging.info( f"final dataset - pos imgs - {len(list(paths.list_images(pos_path)))}" )
        logging.info( f"final dataset - neg imgs - {len(list(paths.list_images(neg_path)))}" )
