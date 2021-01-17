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
      
  /Dataset {USER_PROVIDED}
    /Unlabeled
      im1.png
      im2.png
      
  /Labeled
    /labeled_positive
    /labeled_negative
    
  /to_be_labeled
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
references = f'python deliver_ssl_candidates.py --DATA_PATH {DATA_PATH} --encoder {MODEL_PATH}  --num_workers 2 --batch_size 32 --num_candidates 50'
os.system(references) #Now to_be_labeled is populated with candidate images

num_steps = 10
for i in range(num_steps):
    #call the active labeler to run labeling until to_be_labeled is empty
    
