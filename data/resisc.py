import os
import shutil
import pathlib
from imutils import paths
from random import shuffle
import subprocess
from tqdm import tqdm
import gdown 

def download_and_prepare():
    full_path = os.path.join(os.getcwd(), "RESISC45")
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    download_path = os.path.join(os.getcwd(), "RESISC45.rar")
    url = 'https://drive.google.com/uc?id=14zEhqi9mczZaLEb33TQuKbhmurn2ClGL&export=download'
    output = 'RESISC45.rar'
    if not os.path.isfile(download_path):
      gdown.download(url, output, quiet=False)
    commands = [
    "unrar x {}".format(download_path), 
    "mv NWPU-RESISC45/ RESISC45/",
    "rm {}".format(download_path)]

    os.system( " && ".join(commands))

    folder = 'Dataset/Unlabelled'
    if os.path.exists(folder):
        shutil.rmtree(folder)

    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    for i in tqdm(paths.list_images(full_path)):
        shutil.copy(i, os.path.join(folder, i.split('/')[-1]))
    print('Downloaded and prepared RESISC-45')


def resisc_annotate(image_paths, num_images, already_labelled, positive_class, labelled_dir="Dataset/Labeled", val = False):
  if not val:
    num_labelled = 0
    shuffle(image_paths)
    for image in image_paths:
      if image not in already_labelled:
        num_labelled += 1
        already_labelled.append(image)
        if image.split('/')[-1].split('_')[0] == positive_class:
          shutil.copy(image, os.path.join(labelled_dir,'positive',image.split('/')[-1]))
        else:
          shutil.copy(image, os.path.join(labelled_dir,'negative',image.split('/')[-1]))
      if num_labelled==num_images:
        break
    return already_labelled
  else:
    num_labelled = 0
    num_images_pos = num_images // 2
    all_pos_images = [image for image in image_paths if image.split('/')[-1].split('_')[0] == positive_class]
    all_neg_images = [image for image in image_paths if not image.split('/')[-1].split('_')[0] == positive_class]
    shuffle(all_pos_images)
    pos_images = [all_pos_images[i] for i in range(num_images_pos)]
    neg_images = [all_neg_images[i] for i in range(num_images_pos)]
        
    for image in pos_images + neg_images:
      if image not in already_labelled:
        num_labelled += 1
        already_labelled.append(image)
        if image.split('/')[-1].split('_')[0] == positive_class:
          shutil.copy(image, os.path.join(labelled_dir,'positive',image.split('/')[-1]))
        else:
          shutil.copy(image, os.path.join(labelled_dir,'negative',image.split('/')[-1]))
      if num_labelled==num_images:
        break
    return already_labelled



