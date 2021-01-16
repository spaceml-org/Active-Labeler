/original
  
/data
  
  /to_be_labeled  
    im1.png
    im2.png
    
  /swipe-labeler-data
      /labeled-pos
      /labeled-neg


def get_candidates(classifier_checkpoint, DATA_FOLDER, batch_size, strategy_name = 'uncertain'):
    '''
    input:
    classifier_checkpoint: path to the classification model
    DATA_FOLDER: absolute path to folder containing all the unlabeled images (i.e. copy_original)
    batch_size: amount of candidates to return, must be less than amount of images in unlabeled folder or will return max possible
    strategy_name: name of strategy
    
    output:
    candidates: list of absolute file paths of candidate images of length batch_size
    ['/unlabeled/im2.png', '/unlabeled/im2.png']
    ''' 
    
    
def train_model(classifier_checkpoint, TRAINING_FOLDER):
    '''
    input:
    classifier_checkpoint: path to the classification model
    TRAINING_FOLDER: absolute path to folder containing all labeled images (i.e. path to swipe-labeler-data)
    
    output:
    model_checkpoint: path to the classification model
    ''' 
