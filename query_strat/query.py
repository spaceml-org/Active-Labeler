
import os 
import numpy as np
from torchvision import transforms
from operator import itemgetter
import torch

from data.custom_datasets import AL_Dataset
import query_strat.query_strategies as query_strategies

def get_low_conf_unlabeled_batched(model, image_paths, already_labelled, **al_kwargs):

  strategy = al_kwargs['strategy']
  num_labelled = al_kwargs['num_labelled']
  limit = al_kwargs['limit']

  confidences =  []
  unlabeled_imgs = [os.path.expanduser(img) for img in image_paths if img not in already_labelled]
  t = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0, 0, 0),(1, 1, 1))])
 
  dataset = AL_Dataset(unlabeled_imgs, limit, t)
  data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=4, batch_size=64) #add num workers arg

  confidences = {'conf_vals': [],
                 'loc' : []}

  with torch.no_grad():
    for _, data in enumerate(data_loader):
      image, loc = data
      outputs = model(image.to('cuda'))
      outputs = outputs.detach().cpu().numpy().reshape(-1).tolist()
      confidences['loc'].extend(loc)
      confidences['conf_vals'].extend(outputs)

  confidences['conf_vals'] = np.array(confidences['conf_vals'])

  query_results = getattr(query_strategies, strategy)(confidences, num_labelled)

  return query_results