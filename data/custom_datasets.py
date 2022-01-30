from torch.utils.data import Dataset
from PIL import Image
from random import shuffle
import global_constants as GConst
from imutils import paths
from torchvision import transforms
import os 

class AL_Dataset(Dataset):

  def __init__(self, unlabeled_imgs,  limit, transform = None):
    self.unlabeled_imgs = unlabeled_imgs
    self.transform =  transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0, 0, 0),(1, 1, 1))])
    if limit == -1:
      print("Getting confidences for entire unlabelled dataset")
    else:
      print(f"Getting Confidences from random {limit} data")
      shuffle(self.unlabeled_imgs)
      self.unlabeled_imgs = self.unlabeled_imgs[:limit]
    self.transform = transform

  def __len__(self):
    return len(self.unlabeled_imgs)

  def __getitem__(self, index):
    img_path = self.unlabeled_imgs[index]
    img = Image.open(img_path).convert('RGB')
    
    if self.transform:
      img = self.transform(img)
    return img, img_path

class RESISC_Eval(Dataset):
    def __init__(self, path, positive_class=None) -> None:
        self.images = list(paths.list_images(path))
        self.positive_class  = positive_class
        self.transform = transforms.Compose([
                          transforms.Resize((224,224)),
                          transforms.ToTensor(),
                          transforms.Normalize((0, 0, 0),(1, 1, 1))])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.unlabeled_imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = 1 if self.positive_class in img_path else 0

        return img, label