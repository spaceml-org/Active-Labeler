#Folder with Images
'''
--OFFICIAL DIRECTORY STRUCTURE--

/Local_Directory
  /Swipe-Labeler
  
  /SpaceForceDataSearch
  
  /Active-Labeller
    
  /Models
    /SSL
      SIMCLR_SSL_ssl.ckpt
    /FineTune
      Finetune_ft.ckpt
      
  /Dataset {USER_PROVIDED - DATA_PATH}
    /Unlabeled
      im1.png
      im2.png
      
  /Labeled
    /Labeled_Positive
    /Labeled_Negative
    
  /To_Be_Labeled
    im3.png
    im3.png

'''
#To Do (J): Change folder structure of swipe-labeler to match folder structure above

import os
from argparse import ArgumentParser

def cli_main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='how many items to label')
    parser.add_argument('--DATA_PATH', type=str, help='path to data')
    
    DATA_PATH = args.DATA_PATH
    batch_size = args.batch_size
    
    ssl_train = f'python SpaceForceDataSearch/ssl_dali_distrib.py --DATA_PATH {DATA_PATH} --encoder minicnn32  --num_workers 2 --log_name ssl --epochs 5 --batch_size 32  && wait'
    os.system(ssl_train)
    MODEL_PATH = "./Models/SSL/SIMCLR_SSL_ssl.ckpt"
    print('SSL TRAINED________')
    #get reference images using model
    #to do (Done) (R): make this file
    references = f'python Active-Labeller/deliver_candidates.py --DATA_PATH {DATA_PATH} --encoder {MODEL_PATH}  --num_workers 2 --batch_size 32 --num_candidates 200  && wait'
    os.system(references) #Now to_be_labeled is populated with candidate images
    TO_LABEL = "./To_Be_Labeled"
    print('CANDIDATES MOVED________')
    num_steps = 3
    for i in range(num_steps):
        #call the active labeler to run labeling until to_be_labeled is empty
        #to do (J): change swipe-labeler to use python app.run and add TO_LABEL argument
        #https://stackoverflow.com/questions/48346025/how-to-pass-an-arbitrary-argument-to-flask-through-app-run
        labeler = f'python ./Swipe-Labeler/api/api.py --batch_size 3 --to_be_labeled {TO_LABEL} && wait'
        os.system(labeler)
        print('LABELS DONE BEING LABELED________')
        #train a model
        train_model = f'python SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH "./Labeled" --encoder {MODEL_PATH}  --num_workers 2 --log_name ft --epochs 5 --batch_size 32  && wait'
        os.system(train_model)
        MODEL_PATH = "./Models/FineTune/FineTune_ft.ckpt"
        print('FINETUNING DONE________')
        
        get_more_candidates = f'python deliver_candidates.py --DATA_PATH {DATA_PATH} --encoder {MODEL_PATH}  --num_workers 2 --batch_size 32 --num_candidates 200'
        os.system(get_more_candidates)
        print('GOT MORE CANDIDATES DONE________')
        
   
if __name__ == '__main__':
    cli_main()
