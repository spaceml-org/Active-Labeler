import copy
import random
import time

import yaml
import torch

from models import SIMCLR, SIMSIAM
from Linear_models import SSLEvaluator
from Linear_models_one_layer import SSLEvaluatorOneLayer
from ClassifierModel import ClassifierModel

class TrainModels:
    """This class is used to combine the SSL model’s encoder with linear
    classification layers and fine-tune the model on new dataset."""

    def __init__(self, config_path, model_path, unlabled_dataset_path):
        """
        Combines the SSL model’s encoder with linear classification layers initialises
        the training paramenters.

        Keyword arguments
        config_path -- File path of the yaml configuration file that has the model’s training details
        model_path -- Location where you want to save your model
        unlabled_dataset_path -- File path of your unlabeled data in a pytorch ImageFolder structure.
        """

        def load_config(config_path):
            """Loads the yaml config file"""
            with open(config_path) as file:
                config = yaml.safe_load(file)
            return config

        self.model_path = model_path
        self.parameters = load_config(config_path)
        random.seed(self.parameters["miscellaneous"]["seed"])
        self.log_count = 0
        # Load the encoder of the SSL model
        if self.parameters["encoder"]["encoder_type"] == "SIMCLR":
            self.encoder = SIMCLR.SIMCLR.load_from_checkpoint(
                self.parameters["encoder"]["encoder_path"],
                DATA_PATH=unlabled_dataset_path,
            ).encoder
        elif self.parameters["encoder"]["encoder_type"] == "SIMSIAM":
            self.encoder = SIMSIAM.SIMSIAM.load_from_checkpoint(
                self.parameters["encoder"]["encoder_path"],
                DATA_PATH=unlabled_dataset_path,
            ).encoder
        # Initialise the linear classifier
        if self.parameters["classifier"]["classifier_type"] == "SSLEvaluator":
            self.linear_model = SSLEvaluator(
                n_input=self.parameters["encoder"]["e_embedding_size"],
                n_classes=self.parameters["classifier"]["c_num_classes"],
                p=self.parameters["classifier"]["c_dropout"],
                n_hidden=self.parameters["classifier"]["c_hidden_dim"],
            )
        elif self.parameters["classifier"]["classifier_type"] == "SSLEvaluatorOneLayer":
            self.linear_model = SSLEvaluatorOneLayer(
                n_input=self.parameters["encoder"]["e_embedding_size"],
                n_classes=self.parameters["classifier"]["c_num_classes"],
                p=self.parameters["classifier"]["c_dropout"],
                n_hidden=self.parameters["classifier"]["c_hidden_dim"],
            )
        else:
            raise NameError("Not Implemented")

        # Encoder
        self.lr_encoder = float(self.parameters["encoder"]["e_lr"])
        # Classifier
        self.num_classes = self.parameters["classifier"]["c_num_classes"]
        self.hidden_dim = self.parameters["classifier"]["c_hidden_dim"]
        self.lr_linear = float(self.parameters["classifier"]["c_linear_lr"])
        self.dropout = self.parameters["classifier"]["c_dropout"]
        self.scheduler_type = self.parameters["classifier"]["c_scheduler_type"]
        self.gamma = self.parameters["classifier"]["c_gamma"]
        self.decay_epochs = self.parameters["classifier"]["c_decay_epochs"]
        self.weight_decay = float(self.parameters["classifier"]["c_weight_decay"])
        self.final_lr = float(self.parameters["classifier"]["c_final_lr"])
        self.momentum = float(self.parameters["classifier"]["c_momentum"])
        self.weights = None
        # Training
        self.epochs = self.parameters["training"]["epochs"]
        self.optimizer = None
        self.scheduler = None
        # Device
        self.device = self.parameters["miscellaneous"]["device"]
        # Initialise the model
        self.model = ClassifierModel(self.device, self.encoder, self.linear_model)
        # Loss function
        self.criterion = torch.nn.BCELoss()
        # Load the model parameters

    def train_all(self, training_dataset, validation_dataset):
        """
        Unfreezes the encoder and fine-tunes the entire model on the new dataset

        Keyword arguments
        training_dataset -- A dataloader with the training data
        validation_dataset -- A dataloader with the validation data
        """
        # Setup
        self.model.unfreeze_encoder()
        dataloaders = {"train": training_dataset, "val": validation_dataset}

        self.optimizer = torch.optim.SGD(
            [
                {"params": self.model.encoder.parameters(), "lr": self.lr_encoder},
                {"params": self.model.linear_model.parameters(), "lr": self.lr_linear},
            ],
            momentum=self.momentum,
        )
        if self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.decay_epochs, gamma=self.gamma
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.epochs, eta_min=self.final_lr
            )
        best_model_wts = self.pytorch_training(dataloaders)
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        self.log_count += 1

    def train_linear(self, training_dataset, validation_dataset):
        """
        Freezes the encoder and fine-tunes just the linear-layers

        Keyword arguments
        training_dataset -- A dataloader with the training data
        validation_dataset -- A dataloader with the validation data
        """
        # Setup
        self.model.freeze_encoder()
        dataloaders = {"train": training_dataset, "val": validation_dataset}
        self.optimizer = torch.optim.SGD(
            [
                {"params": self.model.encoder.parameters(), "lr": 0},
                {"params": self.model.linear_model.parameters(), "lr": self.lr_linear},
            ],
            momentum=self.momentum,
        )
        if self.scheduler_type == "step":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.decay_epochs, gamma=self.gamma
            )
        elif self.scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, self.epochs, eta_min=self.final_lr
            )
        # Training
        best_model_wts = self.pytorch_training(dataloaders)
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        self.log_count += 1

    def pytorch_training(self, dataloaders):
        """
        A pytorch training loop used for training by the train_all and train_linear methods

        Keyword arguments
        dataloaders -- A dictionary that contrains the training dataloader and the validation dataloader
        """
        since = time.time()
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs - 1))
            print("-" * 10)
            for phase in ["train", "val"]:
                if phase == "train":
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.parameters["miscellaneous"]["device"])
                    labels = labels.to(self.parameters["miscellaneous"]["device"])
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = self.model(inputs)
                        labels = torch.unsqueeze(labels, 1)
                        loss = self.criterion(outputs.float(), labels.float())
                        if phase == "train":
                            loss.backward()
                            self.optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(outputs == labels.data)
                if phase == "train":
                    self.scheduler.step()

                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects.double() / len(dataloaders[phase])

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
            print()
        print(
            f"Training complete in {(time.time() - since) // 60}m {(time.time() - since) % 60}s"
        )
        print(f"Best val Acc: {best_acc}")
        return best_model_wts

    def get_model(self):
        """Returns the classification model"""
        return self.model

    def save_model(self):
        """Saves the model to the specified path"""
        torch.save(self.model, self.model_path)
