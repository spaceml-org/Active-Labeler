import os 
import json

from data.swipe_labeler import SwipeLabeller
from pigeon import annotate
from IPython.display import display, Image

from utils import annotate_data

annotations = annotate(
  ['assets/img_example1.jpg', 'assets/img_example2.jpg'],
  options=['cat', 'dog', 'horse'],
  display_fn=lambda filename: display(Image(filename))
)

class Labeler:

    def __init__(self, config) -> None:
        
        self.config = config
        method = self.config['active_learning']['labeler']['method']
        if method == "swipelabeler":
            self.labeler = SwipeLabeller(self.config)
        elif method == 'autolabel':
            self.positive_class = self.config['active_learning']['labeler']['positive_class']
        self.method = method

    def label(self, paths, is_eval = False):

        def call_anno(annotations):
            positive = [x[0] for x in annotations if x[1] == 'positive']
            negative = [x[0] for x in annotations if x[1] == 'negative']
            if not is_eval:
                annotate_data(positive, "positive")
                annotate_data(negative, "negative")
            else:
                annotate_data(positive, "eval_pos")
                annotate_data(negative, "eval_neg")

        if self.method == "swipelabeler":
            self.labeler.label(paths, is_eval= is_eval)
        
        elif self.method == 'pigeon':
            annotations = annotate(
            paths,
            options=['positive', 'negative'],
            display_fn=lambda filename: display(Image(filename))
            )

            call_anno(annotations)

        elif self.method == 'autolabel':
            annotations = list()
            for p in paths:
                if p.split('/')[-2] == self.positive_class: #edit this to what the split is in your dataset
                    annotations.append((p, 'positive'))
                else:
                    annotations.append((p, 'negative'))
            
            call_anno(annotations)

        else:
            raise NotImplementedError("Use one of swipelabeler, autolabel or pigeon")

        return 
    

        

