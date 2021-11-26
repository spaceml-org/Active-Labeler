import os
import shutil
from pathlib import Path
import splitfolders
from termcolor import colored
from enum import Enum

import torch
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from argparse import ArgumentParser

# Internal Package Imports
from models import SIMCLR, SIMSIAM, CLASSIFIER, encoders

# Dictionary of supported Techniques
supported_techniques = {
    "SIMCLR": SIMCLR.SIMCLR,
    "SIMSIAM": SIMSIAM.SIMSIAM,
    "CLASSIFIER": CLASSIFIER.CLASSIFIER,
}


def load_model(args):
    """
    A method to load models via command line. Accepts args, a Namespace python object.
    In the method, we first check if the model is a ckpt file. If it is, try loading the checkpoint. If the checkpoint doesn't load, we will attempt to get only the encoder to load via the specified technique
    If the model is not a .ckpt file, we will load it as an encoder from our list of supported encoders.
    Finally, if it is none of the above, it could be a user specified .pt file to represent the encoder.
    """
    technique = supported_techniques[args.technique]
    model_options = Enum(
        "Models_Implemented", "resnet18 imagenet_resnet18 resnet50 imagenet_resnet50"
    )

    if ".ckpt" in args.model:
        args.checkpoint_path = args.model

        try:
            return technique.load_from_checkpoint(**args.__dict__)
        except:
            print("Trying to return model encoder only...")

            # there may be a more efficient way to find right technique to load
            for previous_technique in supported_techniques.values():
                try:
                    args.encoder = previous_technique.load_from_checkpoint(
                        **args.__dict__
                    ).encoder
                    print(
                        colored(
                            f"Successfully found previous model {previous_technique}",
                            "blue",
                        )
                    )
                    break
                except:
                    continue

    # encoder specified
    elif "minicnn" in args.model:
        # special case to make minicnn output variable output embedding size depending on user arg
        output_size = int("".join(x for x in args.model if x.isdigit()))
        args.encoder = encoders.miniCNN(output_size)
        args.encoder.embedding_size = output_size
    elif args.model == model_options.resnet18.name:
        args.encoder = encoders.resnet18(
            pretrained=False,
            first_conv=True,
            maxpool1=True,
            return_all_feature_maps=False,
        )
        args.encoder.embedding_size = 512
    elif args.model == model_options.imagenet_resnet18.name:
        args.encoder = encoders.resnet18(
            pretrained=True,
            first_conv=True,
            maxpool1=True,
            return_all_feature_maps=False,
        )
        args.encoder.embedding_size = 512
    elif args.model == model_options.resnet50.name:
        args.encoder = encoders.resnet50(
            pretrained=False,
            first_conv=True,
            maxpool1=True,
            return_all_feature_maps=False,
        )
        args.encoder.embedding_size = 2048
    elif args.model == model_options.imagenet_resnet50.name:
        args.encoder = encoders.resnet50(
            pretrained=True,
            first_conv=True,
            maxpool1=True,
            return_all_feature_maps=False,
        )
        args.encoder.embedding_size = 2048

    # try loading just the encoder
    else:
        print("Trying to initialize just the encoder from a pytorch model file (.pt)")
        try:
            args.encoder = torch.load(args.model)
        except:
            raise Exception("Encoder could not be loaded from path")
        try:
            embedding_size = encoder.embedding_size
        except:
            raise Exception(
                "Your model specified needs to tell me its embedding size. I cannot infer output size yet. Do this by specifying a model.embedding_size in your model instance"
            )

    # We are initing from scratch so we need to find out how many classes are in this dataset. This is relevant info for the CLASSIFIER
    args.num_classes = len(ImageFolder(args.DATA_PATH).classes)
    return technique(**args.__dict__)


def cli_main():

    parser = ArgumentParser()
    parser.add_argument(
        "--DATA_PATH", type=str, help="path to folders with images to train on."
    )
    parser.add_argument(
        "--VAL_PATH",
        type=str,
        default=None,
        help="path to validation folders with images",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model to initialize. Can accept model checkpoint or just encoder name from models.py",
    )
    parser.add_argument(
        "--batch_size", default=128, type=int, help="batch size for SSL"
    )
    parser.add_argument(
        "--cpus", default=1, type=int, help="number of cpus to use to fetch data"
    )
    parser.add_argument(
        "--hidden_dim",
        default=128,
        type=int,
        help="hidden dimensions in projection head or classification layer for finetuning",
    )
    parser.add_argument(
        "--epochs", default=400, type=int, help="number of epochs to train model"
    )
    parser.add_argument(
        "--learning_rate", default=1e-3, type=float, help="learning rate for encoder"
    )
    parser.add_argument(
        "--patience",
        default=-1,
        type=int,
        help="automatically cuts off training if validation does not drop for (patience) epochs. Leave blank to have no validation based early stopping.",
    )
    parser.add_argument(
        "--val_split",
        default=0.2,
        type=float,
        help="percent in validation data. Ignored if VAL_PATH specified",
    )
    parser.add_argument(
        "--withhold_split",
        default=0,
        type=float,
        help="decimal from 0-1 representing how much of the training data to withold from either training or validation. Used for experimenting with labels neeeded",
    )
    parser.add_argument(
        "--gpus", default=1, type=int, help="number of gpus to use for training"
    )
    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        help="name of model to log on wandb and locally",
    )
    parser.add_argument(
        "--image_size", default=256, type=int, help="height of square image"
    )
    parser.add_argument(
        "--resize",
        default=False,
        type=bool,
        help="Pre-Resize data to right shape to reduce cuda memory requirements of reading large images",
    )
    parser.add_argument(
        "--technique", default=None, type=str, help="SIMCLR, SIMSIAM or CLASSIFIER"
    )
    parser.add_argument(
        "--seed", default=1729, type=int, help="random seed for run for reproducibility"
    )

    # add ability to parse unknown args
    args, _ = parser.parse_known_args()
    technique = supported_techniques[args.technique]
    args, _ = technique.add_model_specific_args(parser).parse_known_args()

    # logging
    wandb_logger = None
    log_name = args.technique + "_" + args.log_name + ".ckpt"
    if log_name is not None:
        wandb_logger = WandbLogger(name=log_name, project="Curator")

    # resize images here
    if args.resize:
        # implement resize and modify args.DATA_PATH accordingly
        pass

    # Splitting Data into train and validation
    if (
        not (
            os.path.isdir(f"{args.DATA_PATH}/train")
            and os.path.isdir(f"{args.DATA_PATH}/val")
        )
        and args.val_split != 0
        and args.VAL_PATH is None
    ):
        print(
            colored(
                f"Automatically splitting data into train and validation data...",
                "blue",
            )
        )
        shutil.rmtree(f"./split_data_{log_name[:-5]}", ignore_errors=True)
        splitfolders.ratio(
            args.DATA_PATH,
            output=f"./split_data_{log_name[:-5]}",
            ratio=(
                1 - args.val_split - args.withhold_split,
                args.val_split,
                args.withhold_split,
            ),
            seed=args.seed,
        )
        args.DATA_PATH = f"./split_data_{log_name[:-5]}/train"
        args.VAL_PATH = f"./split_data_{log_name[:-5]}/val"

    model = load_model(args)
    print(colored("Model architecture successfully loaded", "blue"))

    cbs = []
    backend = "ddp"

    if args.patience > 0:
        cb = EarlyStopping("val_loss", patience=args.patience)
        cbs.append(cb)

    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=20,
        callbacks=cbs,
        distributed_backend=f"{backend}" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        logger=wandb_logger,
        enable_pl_optimizer=True,
    )
    trainer.fit(model)

    Path(f"models/").mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(f"./models/{log_name}")
    print(colored("YOUR MODEL CAN BE ACCESSED AT: ", "blue"), f"./models/{log_name}")


if __name__ == "__main__":
    cli_main()
