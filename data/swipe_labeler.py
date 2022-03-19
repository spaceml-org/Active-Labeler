import os
import json
from turtle import pos
import global_constants as GConst
from utils import get_num_files
from imutils import paths
import shutil
from tqdm import tqdm

class SwipeLabeller:
    def __init__(self, config):
        print(os.getcwd())
        self.config = config
        self.unlabelled_path = GConst.UNLABELLED_DIR
        self.labelled_path =  GConst.LABELLED_DIR
        self.eval_path = GConst.EVAL_DIR
        self.eval_pos_path = os.path.join(GConst.EVAL_DIR,'positive')
        self.eval_neg_path = os.path.join(GConst.EVAL_DIR, 'negative')
        self.positive_path = os.path.join(GConst.LABELLED_DIR,'positive')
        self.negative_path = os.path.join(GConst.LABELLED_DIR,'negative')
        self.unsure_path = GConst.UNSURE_DIR
        print("Swipe Labeler Initialised")

        print(f"\n {len(list(paths.list_images(self.unlabelled_path)))} images to label.")

        # unsure_path = self.parameters["swipe_labeler"]["unsure_path"]

    def move_images_tbl(self, imgs):
        print('Moving images to To Be Labelled folder')
        for img_path in tqdm(imgs):
            shutil.copy(img_path, GConst.TBL_DIR)

    def label(self, imgs, is_eval = False):
        
        self.move_images_tbl(imgs)
        num_unlabelled = len(os.listdir(GConst.TBL_DIR))
        batch_size = min(num_unlabelled, self.config['active_learner']['num_labelled'])

        swipe_dir = GConst.SWIPE_LABELLER_DIR
        swipe_log = " > swipelabeler.log 2>&1"
        #swipe_log = f"> {os.path.join(self.parameters['runtime_path'], 'swipelabeler.log')}"
        pos_path = self.positive_path if not is_eval else self.eval_pos_path
        neg_path = self.negative_path if not is_eval else self.eval_neg_path

        label = f"python3 {swipe_dir} --path_for_unlabeled='{GConst.TBL_DIR}' --path_for_pos_labels='{pos_path}' --path_for_neg_labels='{neg_path}' --path_for_unsure_labels='{self.unsure_path}' --batch_size={batch_size} {swipe_log}"
        
        print(label)
        ossys = os.system(label)
        print(f"swipe labeler exit code {ossys}")
        print(f" Labelling complete, {num_unlabelled} files labelled")
        shutil.rmtree(GConst.TBL_DIR)
        os.makedirs(GConst.TBL_DIR)
        print(f"Total Labelled : {get_num_files('labelled') if not is_eval else get_num_files('eval_labelled')} \n Positive : {len(os.listdir(pos_path))} Negative : {len(os.listdir(neg_path))}")