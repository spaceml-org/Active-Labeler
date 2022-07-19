import os 
import numpy as np
from torchvision import transforms
from operator import itemgetter
from tqdm import tqdm
import torch
from query_strat.diversity_sampling import pick_top_n, iterative_proximity_sampling, clustering_sampling, random_sampling
from query_strat.query_strategies import entropy_based, margin_based, least_confidence
from data.custom_datasets import AL_Dataset

def get_low_conf_unlabeled_batched(model, image_paths, already_labeled, train_kwargs, **al_kwargs):

  strategy = al_kwargs['strategy']
  diversity_sampling = al_kwargs['diversity_sampling']
  num_labeled = al_kwargs['num_labeled']
  limit = al_kwargs['limit']

  confidences =  []
  unlabeled_imgs = [os.path.expanduser(img) for img in image_paths if img not in already_labeled]
  t = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()])
  
  dataset = AL_Dataset(unlabeled_imgs, limit, t)
  unlabeled_loader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=4, batch_size=64) #add num workers arg

  confidences = {'conf_vals': [],
                 'loc' : []}

  batch_bar = tqdm(total=len(unlabeled_loader), dynamic_ncols=True, leave=False, position=0, desc='Get Most Uncertain Samples') 
  model.eval()
  all_embeddings = []

  with torch.no_grad():
    for _, (image, loc) in enumerate(unlabeled_loader):
      
      outputs, embeddings = model(image.to('cuda'), True)

      outputs = outputs.detach().cpu().numpy()
      embeddings = embeddings.detach().cpu().numpy()

      all_embeddings.append(embeddings)
      confidences['loc'].extend(loc)
      
      confidences['conf_vals'].append(outputs)

      batch_bar.update()

  batch_bar.close()

  all_embeddings = np.concatenate(all_embeddings)



  confidences['conf_vals'] = np.concatenate(confidences['conf_vals'])

  confidences['loc'] = np.array(confidences['loc'])

  if strategy == 'margin_based':
    uncertainty_scores = margin_based(confidences)
  elif strategy == 'least_confidence':
    uncertainty_scores = least_confidence(confidences)
  elif strategy == 'entropy_based':
    uncertainty_scores = entropy_based(confidences)
  elif strategy == 'random_sampling':
    print("You are using random sampling for your uncertainty criteria.")
  else:
    assert False,"AL Strategy not present"
    
  # close to 1 is more uncertain
  # now take uncertainties and use it to perform diversity sampling.

  if diversity_sampling == "pick_top_n":
    selected_filepaths = pick_top_n(uncertainty_scores, confidences['loc'], num_labeled)
  elif diversity_sampling == "iterative_proximity_sampling":
    selected_filepaths = iterative_proximity_sampling(uncertainty_scores, confidences['loc'], num_labeled, all_embeddings)
  elif diversity_sampling == "clustering_sampling":
    selected_filepaths = clustering_sampling(uncertainty_scores, confidences['loc'], num_labeled, all_embeddings)
  elif diversity_sampling == "random_sampling":
    selected_filepaths = random_sampling(confidences['loc'], num_labeled)
  else:
    assert False, "Diversity Strategy not present"

  #making sure there are no duplicates. 
  assert len(selected_filepaths) == len(set(selected_filepaths)), "Duplicates found"

  return selected_filepaths