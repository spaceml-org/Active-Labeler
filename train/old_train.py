import torch
import torchvision 
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from random import shuffle
import datetime 
import re 

def train_model(model, train_datapath, eval_datapath,epochs, select_per_epoch, loss,opt):
  t = transforms.Compose([transforms.ToTensor()])

  loss_fn = nn.BCELoss()
  opt = torch.optim.SGD(list(model.fc.parameters())+list(model.hidden.parameters()),lr=0.01)
  train_dataset = ImageFolder(train_datapath)
  train_imgs = train_dataset.imgs
  eval_dataset = ImageFolder(eval_datapath)
  for epoch in range(epochs):
    print("Epoch :",str(epoch))
    current = 0
    shuffle(train_imgs)
    
    # make a subset of data to use in this epoch
    # with an equal number of items from each label

    pos = [img for img in train_imgs if img[1]==1]
    neg = [img for img in train_imgs if img[1]==0]
    epoch_data = pos[:select_per_epoch]
    epoch_data+= neg[:select_per_epoch] 
    shuffle(epoch_data)

    #model training
    for item in epoch_data:
      features = t(Image.open(item[0])).unsqueeze(0)
      target = item[1]
      target = torch.FloatTensor([target]).to('cuda')
      model.zero_grad()

      log_probs = model(features.to('cuda')) 
      loss = loss_fn(log_probs[0], target)
      loss.backward()
      opt.step()
    fscore, auc = evaluate_model(model.to('cuda'), eval_dataset)
    fscore = round(fscore,3)
    auc = round(auc,3)
    print("Results: F1score {}, AUC {}".format(fscore, auc))

  # save model to path that is alphanumeric and includes number of items and accuracies in filename
  timestamp = re.sub('\.[0-9]*','_',str(datetime.datetime.now())).replace(" ", "_").replace("-", "").replace(":","")
  training_size = "_"+str(len(train_imgs))
  accuracies = str(fscore)+"_"+str(auc)
  model_path = "models/"+timestamp+accuracies+training_size+".params"
  torch.save(model.state_dict(), model_path)
  return model_path

def evaluate_model(model, evaluation_data, threshold):
  t = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])  
  pos_confs = []
  neg_confs = []

  true_pos = 0.0
  false_pos = 0.0
  false_neg = 0.0

  with torch.no_grad():
    for item in evaluation_data.imgs:
      features = t(Image.open(item[0])).unsqueeze(0)
      label = item[1]
      log_probs = model(features.to('cuda'))

      prob_related = log_probs.detach().cpu().data.tolist()[0][0]
      if label==1:
        pos_confs.append(prob_related)
        if prob_related > threshold:
          true_pos += 1.0
        else:
          false_neg += 1.0
      else:
          #negative class
          neg_confs.append(prob_related)
          if prob_related > threshold:
            false_pos += 1.0
      
  #Get FScore
  if true_pos == 0.0:
    fscore = 0.0
  else:
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    fscore = 2*precision*recall/(precision+recall)

  #Get AUC
  neg_confs.sort()
  total_greater = 0 # count of how many total have higher confidence
  for conf in pos_confs:
    for conf2 in neg_confs:
      if conf < conf2:
        break
      else:
        total_greater += 1
  denom = len(neg_confs) * len(pos_confs) 
  auc = total_greater / denom
  return[fscore, auc]