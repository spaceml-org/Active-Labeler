import sys
import os
import pandas as pd
from imutils import paths
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import optim, cuda
import warnings
import numpy as np
import global_constants as GConst

warnings.filterwarnings("ignore")
sys.path.append("{}/external_lib/SSL/".format(os.getcwd()))

from utils import (
    get_num_files,
    load_config,
    load_model,
    load_opt_loss,
    initialise_data_dir,
)
from train.train_model import train_model_vanilla
from query_strat.query import get_low_conf_unlabeled_batched
from data.indexer import Indexer
from data.label import Labeler
import data.tfds as tfds_support

from utils import load_config, load_model, load_opt_loss
from query_strat.query import get_low_conf_unlabeled_batched
from data.sample import stratified_sampling


class Pipeline:
    def __init__(self, config_path) -> None:
        self.config = load_config(config_path)
        self.model_kwargs = self.config["model"]

        # self.optim, self.loss = load_opt_loss(self.model, self.config)
        self.already_labelled = list()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0, 0, 0), (1, 1, 1)),
            ]
        )

        self.labeler = Labeler(self.config)

    def main(self):
        config = self.config
        if config["data"]["dataset"] == "tfds":
            dataset_name = config["data"]["dataset_name"]
            dataset_path = os.path.join(os.getcwd(), dataset_name)
            print("Dataset:", dataset_name)
            tfds_prepare = tfds_support.PrepareData(dataset_name, config)
            tfds_prepare.download_and_prepare()

            # Initialising data by annotating labeled
            unlabeled_images = list(paths.list_images(GConst.UNLABELED_DIR))
            self.already_labeled = tfds_support.tfds_annotate(
                unlabeled_images,
                1500,
                self.already_labeled,
                labeled_dir=GConst.LABELED_DIR,
            )

            # Create validation and test dataset objects
            val_dataset = ImageFolder(GConst.VAL_DIR, transform=self.transform)
            test_dataset = ImageFolder(GConst.TEST_DIR, transform=self.transform)

            al_config = config["active_learner"]
            al_kwargs = dict(
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                strategy=al_config["strategy"],
                diversity_sampling=al_config["diversity_sampling"],
                num_iters=al_config["iterations"],
                num_labeled=al_config["num_labeled"],
                limit=al_config["limit"],
            )

            self.train_al(unlabeled_images, **al_kwargs)

        elif config["data"]["dataset"] == "csv":
            # config = airplanes, harbor, cars.
            # img_path   label
            # abc.jpg    airplanes
            # abc.jpg    harbor
            # 1. add to config all classes for annotation
            # 2. create a folder for each class available in labelled/ eval folders instead of current pos neg
            # 3. take query csv, take all possible classes available in query csv and copy to GConst.LABELLED_DIR
            # 4. do faiss/random sampling and initialsie set and copy over to respective class wise folder.
            # csv structure  - , classes

            # tfds flow -
            # 1. give name of dataset, init just sets name and takes config
            # 2. call download_and_prepare -
            #   does tf load of dataset first.
            #   if the folder already doesnt exist, calls tfds_io.
            #      tfds io takes the entire ds, writes files to unlabeled(train), valid and test based on % split
            #   val and test is set here. no changes made to it. torch dataset is created after that
            #   labeled is left empty, and init + annotate is done in a separate tfds annotate step.
            # tfds annotate step takes image paths, copies one image of each class to their train folder, then takes a small sample and copies the whole thing

            self.df = pd.read_csv(config["data"]["path"])
            initialise_data_dir(config)
            df = self.df.copy()
            print("Data Directory Initialised")
            labeled_df = df[df["label"].isin(config["data"]["classes"])]
            unlabeled_images = df[df["label"].isna()]
            num_labelled = config["active_learner"]["num_labelled"]
            self.preindex = self.config["active_learner"]["preindex"]
            print("Preindex:", self.preindex)
            if self.preindex:
                model = load_model(**self.model_kwargs)
                self.index = Indexer(
                    unlabeled_images, model, img_size=224, index_path=None
                )
                faiss_init_imgs = list()
                for label in config["data"]["classes"]:
                    print("Indexing Label:", label)
                    query = labeled_df[labeled_df["label"] == label][
                        GConst.IMAGE_PATH_COL
                    ].values[0]
                    similar_imgs = self.index.process_image(
                        query, n_neighbors=num_labelled * 2
                    )
                    faiss_init_imgs.extend(similar_imgs)

                faiss_annotate = self.labeler.label(faiss_init_imgs, fetch_paths=True)
                faiss_annotate = pd.DataFrame(
                    faiss_annotate, columns=[GConst.IMAGE_PATH_COL, "label"]
                )
                print("Labeled DF Before : ", labeled_df.shape)
                labeled_df = labeled_df.append(faiss_annotate)
                print("Labeled DF After : ", labeled_df.shape)

            else:
                print("Number of unlabelled Images:", len(unlabeled_images))
                random_init_imgs = unlabeled_images.sample(n = (num_labelled * 2))[
                    GConst.IMAGE_PATH_COL
                ].values
                random_annotate = self.labeler.label(random_init_imgs, fetch_paths=True)
                random_annotate = pd.DataFrame(
                    random_annotate, columns=[GConst.IMAGE_PATH_COL, "label"]
                )
                print("Labeled DF Before : ", labeled_df.shape)
                labeled_df = labeled_df.append(random_annotate)
                print("Labeled DF After : ", labeled_df.shape)

            self.already_labeled.extend(labeled_df[GConst.IMAGE_PATH_COL].values)
            labeled_df = stratified_sampling(
                labeled_df,
                split_ratio=self.config["active_learner"]["init_split_ratio"],
            )
            train_df = labeled_df[labeled_df["stratified_status"] == "train"]
            val_df = labeled_df[labeled_df["stratified_status"] == "val"]
            assert len(val_df["label"].unique()) == len(
                train_df["label"].unique()
            ), "Val and train have inconsistent labels. Train : {} Val : ".format(
                train_df["label"].unique(), val_df["label"].unique()
            )

            self.labeler.annotate_paths(train_df, is_eval=False)
            self.labeler.annotate_paths(val_df, is_eval=True)

            val_dataset = ImageFolder(GConst.VAL_DIR, transform=self.transform)

            # swipe_labeler -> label random set of data -> labelled pos/neg. Returns paths labelled
            print("Total annotated valset : {}".format(get_num_files("Dataset/Val")))
            print("Total Labeled Data: {}".format(get_num_files("Dataset/Labeled")))

            al_config = config["active_learner"]
            al_kwargs = dict(
                val_dataset=val_dataset,
                strategy=al_config["strategy"],
                diversity_sampling=al_config["diversity_sampling"],
                num_iters=al_config["iterations"],
                num_labeled=al_config["num_labeled"],
                limit=al_config["limit"],
            )
            self.train_al(unlabeled_images, **al_kwargs)

    def train_al(self, unlabeled_images, **al_kwargs):
        iter1 = 0
        val_dataset = al_kwargs["val_dataset"]
        test_dataset = al_kwargs.get("test_dataset", None)
        num_iters = al_kwargs["num_iters"]

        train_config = self.config["train"]
        file1 = open(f"logs/{GConst.start_name}_{GConst.diversity_name}.txt", "a")
        file1.write(f"{GConst.start_name}__{GConst.diversity_name}\n")
        file1.close()

        while iter1 < num_iters:
            print(f"-------------------{iter1 +1}----------------------")
            iter1 += 1
            model = load_model(**self.model_kwargs)
            optimizer, loss = load_opt_loss(model, self.config)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.925)
            scaler = cuda.amp.GradScaler()

            train_kwargs = dict(
                epochs=train_config["epochs"],
                opt=optimizer,
                loss_fn=loss,
                batch_size=train_config["batch_size"],
                scheduler=scheduler,
                scaler=scaler,
            )

            train_model_vanilla(
                model,
                GConst.LABELED_DIR,
                iter1,
                val_dataset,
                test_dataset,
                **train_kwargs,
            )
            low_confs = get_low_conf_unlabeled_batched(
                model, unlabeled_images, self.already_labeled, train_kwargs, **al_kwargs
            )
            for image in low_confs:
                if image not in self.already_labeled:
                    self.already_labeled.append(image)
                    label = image.split("/")[-1].split("_")[0]
                    shutil.copy(
                        image,
                        os.path.join(GConst.LABELED_DIR, label, image.split("/")[-1]),
                    )
            