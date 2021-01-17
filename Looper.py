#Folder with Images
'''
--OFFICIAL DIRECTORY STRUCTURE--

/Local_Directory
  /Active Labeler Repo
    api.py
    ...
    
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

DATA_PATH = './Dataset' #USER_PROVIDED


ssl_train = f'python /content/SpaceForceDataSearch/ssl_dali_distrib.py --DATA_PATH {DATA_PATH} --encoder minicnn32  --num_workers 2 --log_name ssl --epochs 5 --batch_size 32'

#exec this command 
os.system(ssl_train)
MODEL_PATH = "./Models/SSL/SIMCLR_SSL_ssl.ckpt"

#get reference images using model
#to do (R): make this file
references = f'python deliver_candidates.py --DATA_PATH {DATA_PATH} --encoder {MODEL_PATH}  --num_workers 2 --batch_size 32 --num_candidates 200'
os.system(references) #Now to_be_labeled is populated with candidate images
TO_LABEL = "./to_be_labeled"

num_steps = 10
for i in range(num_steps):
    #call the active labeler to run labeling until to_be_labeled is empty
    #to do (J): change swipe-labeler to use python app.run and add TO_LABEL argument
    #https://stackoverflow.com/questions/48346025/how-to-pass-an-arbitrary-argument-to-flask-through-app-run
    labeler = f'python run_labeler.py --TO_LABEL {TO_LABEL} --batch_size 200'
    os.system(labeler)
    
    #train a model
    train_model = f'python /content/SpaceForceDataSearch/finetuner_dali_distrib.py --DATA_PATH "./Labeled" --encoder {MODEL_PATH}  --num_workers 2 --log_name ft --epochs 5 --batch_size 32'
    os.system(train_model)
    MODEL_PATH = "./Models/FineTune/FineTune_ft.ckpt"
    
    get_more_candidates = f'python deliver_candidates.py --DATA_PATH {DATA_PATH} --encoder {MODEL_PATH}  --num_workers 2 --batch_size 32 --num_candidates 200'
    os.system(get_more_candidates)
    
