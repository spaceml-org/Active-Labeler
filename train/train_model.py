import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from random import shuffle
import datetime 
import re
import sklearn.metrics as metrics
import numpy as np
import datetime 
import re
from imutils import paths
import global_constants as GConst

def calculate_metrics(preds, trues, cm=False):

    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]  
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    trues = [int(true.item()) for true in trues]
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    if cm:
      print("Confusion Matrix:")
      print(metrics.confusion_matrix(
            trues,preds,labels=[0,1]
        ))
    fscore = metrics.f1_score(trues, preds,average='binary')
    return (acc * 100), fscore
    

def evaluate_al(model, eval_dataset, loss_fn):
  
  testloader = torch.utils.data.DataLoader(eval_dataset, batch_size=64,
                                      shuffle=True, num_workers=4)

  epoch_loss = []
  preds = []
  trues = []
  for data in testloader:
      images, labels = data
      labels = labels.to(torch.float32)
      labels = labels.reshape((labels.shape[0], 1)).to('cuda')
      outputs = model(images.to('cuda'))
      
      loss = loss_fn(outputs, labels)
      loss = loss.item()
      epoch_loss.append(loss)

      preds.extend(outputs.detach().cpu().numpy())
      trues.extend(labels.detach().cpu().numpy())
  # print(len(preds),len(trues))    
  acc, fscore = calculate_metrics(preds, trues, True)
  print(f"On the whole dataset: \n Accuracy: {acc} \n F1 Score: {fscore}")

def val_model_vanilla(model, eval_dataset, loss_fn):
  
  testloader = torch.utils.data.DataLoader(eval_dataset, batch_size=64,
                                      shuffle=True, num_workers=4)

  epoch_loss = []
  epoch_acc = []
  epoch_f1 = []
  # model.eval()
  # with torch.no_grad():
  for data in testloader:
      images, labels = data
      labels = labels.to(torch.float32)
      labels = labels.reshape((labels.shape[0], 1)).to('cuda')
      outputs = model(images.to('cuda'))
      
      loss = loss_fn(outputs, labels)
      loss = loss.item()
      epoch_loss.append(loss)
      
      acc, fscore = calculate_metrics(outputs, labels)
      epoch_acc.append(acc)
      epoch_f1.append(fscore)
      
  avg_acc = np.mean(epoch_acc) 
  avg_loss = np.mean(epoch_loss)
  avg_fscore = np.mean(epoch_f1)

  return avg_fscore, avg_acc, avg_loss


def train_model_vanilla(model, train_datapath, eval_dataset, val_dataset, **train_kwargs):

  num_epochs = train_kwargs['epochs']
  batch_size = train_kwargs['batch_size']
  opt = train_kwargs['opt']
  loss_fn = train_kwargs['loss_fn']

  t = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.RandomHorizontalFlip(p=0.5),
                          transforms.RandomRotation(15),
                          transforms.RandomCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))
  ])

  
  train_dataset = ImageFolder(train_datapath, transform=t)
  train_imgs = train_dataset.imgs

  trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)


  
  print("Training")
  print('{:<10s}{:>4s}{:>12s}{:>12s}{:>12s}{:>12s}{:>12s}'.format("Epoch", "Train Loss", "Train Acc", "Train F1", "Val Loss", "Val Acc", "Val F1"))

  graph_logs = {}
  graph_logs['val_f1'] = []
  graph_logs['len_data'] = []

  for epoch in range(num_epochs):
    epoch_loss = []
    epoch_acc = []
    epoch_f1 = []

    for _, data in enumerate(trainloader):
        breakpoint()
        inputs, labels = data
        labels = labels.reshape((labels.shape[0], 1))
        labels = labels.to(torch.float32)
        labels = labels.to('cuda')
        opt.zero_grad()

        outputs = model(inputs.to('cuda'))
        loss = loss_fn(outputs, labels)
        epoch_loss.append(loss.item())

        acc, fscore = calculate_metrics(outputs, labels)
        epoch_acc.append(acc)
        epoch_f1.append(fscore)

        loss.backward()
        opt.step()

    avg_acc = np.mean(epoch_acc) 
    avg_loss = np.mean(epoch_loss)
    avg_fscore = np.mean(epoch_f1)

    val_fscore, val_acc, val_loss = val_model_vanilla(model, val_dataset, loss_fn)
    print('{:<10d}{:>4.2f}{:>13.2f}{:>13.2f}{:>13.2f}{:>13.2f}{:>13.2f}'.format(epoch, avg_loss, avg_acc, avg_fscore, val_loss, val_acc, val_fscore))

    print("Validation F1 Score: ",val_fscore,"Total Data Used :",len(list(paths.list_images(GConst.LABELLED_DIR))))
  graph_logs['val_f1'].append(val_fscore)
  graph_logs['len_data'].append(len(list(paths.list_images(GConst.LABELLED_DIR))))
  
  fscore = 0.0
  auc = 0.0
  
  timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
  training_size = str(len(train_imgs))
  accuracies = str(fscore)+"_"+str(auc)
  model_path = "checkpoints/"+timestamp+ accuracies+ "_" + training_size+".params"
  torch.save(model.state_dict(), model_path)
  
  print("Since our model has become confident enough, testing on leftover unlabeled data")
  evaluate_al(model, eval_dataset, loss_fn)

  return model_path, graph_logs


  # #logging the final epoch's val f1 score into the graph logs
  # print("Validation F1 Score: ",val_fscore,"Total Data Used :",len(list(paths.list_images(GConst.LABELLED_DIR))))
  # graph_logs['val_f1'].append(val_fscore)
  # graph_logs['len_data'].append(len(list(paths.list_images(GConst.LABELLED_DIR))))
  
  # fscore = 0.0
  # auc = 0.0
  # timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
  
  # training_size = str(len(train_imgs))
  # accuracies = str(fscore)+"_"+str(auc)
  # model_path = "checkpoints/"+timestamp+accuracies+ "_" + training_size+".params"
  # torch.save(model.state_dict(), model_path)
  # print("Since our model has become confident enough, testing on leftover unlabeled data")
  # evaluate_al(model, eval_dataset)  
  # # evaluate_al(model.to('cuda'),batched_unlabeled_dir,True)
  # # evaluate_al(model.to('cuda'),eval_datapath,True)
  # return model_path, graph_logs
  