import sys
sys.path.insert(0, "./AL_final")
sys.path.insert(0, "./AL_final/ActiveLabeler-main")
sys.path.insert(0, "./AL_final/ActiveLabeler-main/Self-Supervised-Learner")
sys.path.insert(0, "./AL_final/ActiveLabeler-main/ActiveLabelerModels")
import logging
import os.path

import yaml
import random
from argparse import ArgumentParser
import pathlib
import numpy as np
from argparse import ArgumentParser
from pipeline import Pipeline
import shutil

def main():
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--class_name", type=str, help="Positive class name")
    args = parser.parse_args()
    config_path = args.config_path
    class_name = args.class_name

    with open(config_path) as file:
        config = yaml.safe_load(file)

    #set seed
    random.seed(config["seed"])
    np.random.seed(config["seed"])


    #log settings
    if config["verbose"] ==0:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)-8s - %(funcName)-15s - %(message)s', datefmt='%d-%b-%y %H:%M:%S",
            handlers=[
                logging.FileHandler("/content/app.log", mode='a') #TODO logging - give user choice or hardcode ?
            ]
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)-8s - %(funcName)-15s - %(message)s', datefmt='%d-%b-%y %H:%M:%S",
            handlers=[
                logging.FileHandler("/content/app.log", mode='a'),
                logging.StreamHandler()
            ]
        )

    #directories
    for i in [ config["nn"]["unlabled_path"],config["nn"]["labeled_path"],config["nn"]["positive_path"],config["nn"]["negative_path"],config["nn"]["unsure_path"],config["AL_main"]["al_folder"],
               os.path.join(config["AL_main"]["al_folder"],"positive"),os.path.join(config["AL_main"]["al_folder"],"negative"),
               os.path.join(config["AL_main"]["newly_labled_path"], "positive"),
               os.path.join(config["AL_main"]["newly_labled_path"], "negative"),
               os.path.join(self.parameters["AL_main"]["archive_path"], "positive"),
               os.path.join(self.parameters["AL_main"]["archive_path"], "negative"),
               '/'.join(config["annoy"]["annoy_path"].split('/')[:-1]) ]:
      if os.path.exists(i):
        shutil.rmtree(i)
    for i in [ config["nn"]["unlabled_path"],config["nn"]["labeled_path"],config["nn"]["positive_path"],config["nn"]["negative_path"],config["nn"]["unsure_path"],config["AL_main"]["al_folder"],
               os.path.join(config["AL_main"]["al_folder"],"positive"),os.path.join(config["AL_main"]["al_folder"],"negative"),
               os.path.join(config["AL_main"]["newly_labled_path"], "positive"),
               os.path.join(config["AL_main"]["newly_labled_path"], "negative"),
               os.path.join(self.parameters["AL_main"]["archive_path"], "positive"),
               os.path.join(self.parameters["AL_main"]["archive_path"], "negative"),
               '/'.join(config["annoy"]["annoy_path"].split('/')[:-1]) ]:
      pathlib.Path(i).mkdir(parents=True, exist_ok=True)

    # initialize pipeline object
    pipeline = Pipeline(config_path=config_path, class_name=class_name)
    pipeline.main()

if __name__ == '__main__':
    main()




