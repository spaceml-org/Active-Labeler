import torch
import torchvision
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from random import shuffle
import datetime
import re
from tqdm import tqdm
import sklearn.metrics as metrics
import numpy as np
import datetime
import os
import shutil
import re
import time
from imutils import paths
import global_constants as GConst


def val_model_vanilla(
    model, val_dataset, val_loader, loss_fn, batch_size, last_epoch=False
):

    epoch_loss = []
    epoch_acc = []
    epoch_f1 = []
    num_correct_val = 0
    total_loss_val = 0
    true = []
    pred = []
    model.eval()
    batch_bar = tqdm(
        total=len(val_loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )
    for i, (images, labels) in enumerate(val_loader):
        images = images.to("cuda")
        labels = labels.to("cuda")

        with torch.no_grad():
            outputs = model(images)

        loss = loss_fn(outputs, labels)
        loss = loss.item()
        epoch_loss.append(loss)

        outputs = torch.argmax(outputs, axis=1)
        num_correct_val += int((outputs == labels).sum())
        total_loss_val += float(loss)

        if last_epoch:
            true.append(labels.detach().cpu().numpy())
            pred.append(outputs.detach().cpu().numpy())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct_val / ((i + 1) * batch_size)),
            loss="{:.04f}".format(float(total_loss_val / (i + 1))),
        )

        batch_bar.update()
    batch_bar.close()

    valid_acc = num_correct_val / len(val_dataset)
    print(
        "Val Acc: {:.04f}%, Train Loss {:.04f},".format(
            100 * valid_acc, float(total_loss_val / len(val_dataset))
        )
    )

    file1 = open(
        f"logs/{GConst.DATASET_NAME}_{GConst.start_name}_{GConst.diversity_name}.txt",
        "a",
    )
    file1.write(f"Validation Acc: {(100 * valid_acc):.2f}%" + "\n")
    file1.write(f"--------------------------\n")
    file1.close()

    if last_epoch:
        print("CLASSIFICATION REPORT")
        print(metrics.classification_report(np.concatenate(true), np.concatenate(pred)))
        print("CONFUSION MATRIX")
        print(metrics.confusion_matrix(np.concatenate(true), np.concatenate(pred)))

    return num_correct_val, total_loss_val


def train_model_vanilla(
    model, train_datapath, counter, val_dataset=None, test_dataset=None, **train_kwargs
):

    num_epochs = train_kwargs["epochs"]
    batch_size = train_kwargs["batch_size"]
    optimizer = train_kwargs["opt"]
    loss_fn = train_kwargs["loss_fn"]
    scheduler = train_kwargs["scheduler"]
    scaler = train_kwargs["scaler"]

    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(30),
            torchvision.transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    train_dataset = ImageFolder(train_datapath, transform=t)
    train_imgs = train_dataset.imgs

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    file1 = open(
        f"logs/{GConst.DATASET_NAME}_{GConst.start_name}_{GConst.diversity_name}.txt",
        "a",
    )
    file1.write(f"--------Iter Num {counter}---------")
    file1.close()

    print("train_dataset length:", len(train_dataset))
    print("Training")
    graph_logs = {}
    graph_logs["val_f1"] = []
    graph_logs["len_data"] = []
    for epoch in range(num_epochs):
        epoch_loss = []
        num_correct = 0
        total_loss = 0
        model.train()
        batch_bar = tqdm(
            total=len(train_loader),
            dynamic_ncols=True,
            leave=False,
            position=0,
            desc="Train",
        )
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            epoch_loss.append(loss.item())
            outputs = torch.argmax(outputs, axis=1)
            num_correct += int((outputs == labels).sum())
            total_loss += float(loss)

            batch_bar.set_postfix(
                acc="{:.04f}%".format(100 * num_correct / ((i + 1) * batch_size)),
                loss="{:.04f}".format(float(total_loss / (i + 1))),
                num_correct=num_correct,
                lr="{:.04f}".format(float(optimizer.param_groups[0]["lr"])),
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_bar.update()
        batch_bar.close()

        file1 = open(
            f"logs/{GConst.DATASET_NAME}_{GConst.start_name}_{GConst.diversity_name}.txt",
            "a",
        )
        file1.write("\n" + f"Epoch: {epoch+1}" + "\n")
        file1.write(
            f"Training Acc: {100 * num_correct / (len(train_loader) * batch_size):.2f}%"
            + "\n"
        )
        file1.write(f"lr: {optimizer.param_groups[0]['lr']}" + "\n")
        file1.close()

        print(
            "Epoch {}/{}: Train Acc {:.04f}%, Train Loss {:.04f}, Learning Rate {:.04f}".format(
                epoch + 1,
                num_epochs,
                100 * num_correct / (len(train_loader) * batch_size),
                float(total_loss / len(train_loader)),
                float(optimizer.param_groups[0]["lr"]),
            )
        )

        scheduler.step()

        if epoch == num_epochs - 1:
            num_correct_val, total_loss_val = val_model_vanilla(
                model, val_dataset, val_loader, loss_fn, batch_size, last_epoch=True
            )
        else:
            num_correct_val, total_loss_val = val_model_vanilla(
                model, val_dataset, val_loader, loss_fn, batch_size, last_epoch=False
            )
