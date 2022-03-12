import os
import json
import global_constants as GConst
from utils import get_num_files
from imutils import paths

class SwipeLabeller:

    def __init__(self, config):
        print(os.getcwd())
        self.config = config
        self.unlabelled_path = GConst.UNLABELLED_DIR
        self.labelled_path =  GConst.LABELLED_DIR
        self.eval_path = GConst.EVAL_DIR
        self.eval_pos_path = os.path.join(GConst.EVAL_DIR,'positive')
        self.positive_path = os.path.join(GConst.LABELLED_DIR,'positive')
        self.negative_path = os.path.join(GConst.LABELLED_DIR,'negative')
        self.unsure_path = GConst.UNSURE_DIR
        print("Swipe Labeler Initialised")

        print(f"\n {len(list(paths.list_images(self.unlabeled_path)))} images to label.")

        # unsure_path = self.parameters["swipe_labeler"]["unsure_path"]


    def label(self, is_eval = False):
        if is_eval:

        ori_labeled = get_num_files('labelled')
        ori_pos = get_num_files('positive')
        ori_neg = get_num_files('negative')

        batch_size = min(get_num_files('unlabelled'), self.config['active_learner']['num_labelled'])
        swipe_dir = GConst.SWIPE_LABELLER_DIR
        swipe_log = " > swipelabeler.log 2>&1"
        #swipe_log = f"> {os.path.join(self.parameters['runtime_path'], 'swipelabeler.log')}"
        label = f"python3 {swipe_dir} --path_for_unlabeled='{self.unlabelled_path}' --path_for_pos_labels='{self.positive_path}' --path_for_neg_labels='{self.negative_path}' --path_for_unsure_labels='{self.unsure_path}' --batch_size={batch_size} {swipe_log}"
        print(label)
        ossys = os.system(label)
        print(f"swipe labeler exit code {ossys}")

        print(
            f" {get_num_files('labelled') - ori_labeled} labeled: {get_num_files('positive') - ori_pos} Pos {get_num_files('negative') - ori_neg} Neg"
        )

        print(
            f"{get_num_files('labelled')} labeled: {get_num_files('positive')} Pos {get_num_files('negative')} Neg"
        )
        print(f"unlabeled list: {self.unlabelled_list}")
        print(f"labeled list: {self.labelled_list}")
    
    