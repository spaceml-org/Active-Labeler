<img src="https://github.com/spaceml-org/Active-Labeler/blob/main/readme_banner.png" >

<div align="center">
<p align="center">
  Published by <a href="http://spaceml.org/">SpaceML</a> •
  <a href="https://arxiv.org/abs/2012.10610">About SpaceML</a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7%20|%203.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/Cuda-10%20|%2011.0-4dc71f.svg)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
[![Pip Package](https://img.shields.io/badge/Pip%20Package-Coming%20Soon-0073b7.svg)](https://pypi.org/project/pip/)
[![Docker](https://img.shields.io/badge/Docker%20Image-Coming%20Soon-34a0ef.svg)](https://www.docker.com/)
   
[![Google Colab Notebook Example](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/spaceml-org/Self-Supervised-Learner/blob/simsiam/tutorials/PythonColabTutorial_Merced.ipynb)
   
</div>

The Active Labeler is a CLI tool that facilitates labeling datasets with just a SINGLE line of code. The tool comes fully equipped with various Active Learning strategies and other customizable features to suit one’s requirements.

## What is Active Learning?
Bringing about a model’s full potential with limited data is the core concept behind Active Learning. This follows several strategies and other techniques that identify points based on a model’s difficulty in understanding them. 
Some of the strategies present in this tool includes:
-  Random
    * Random function is used to pick examples uniformly from the entire unlabeled dataset.
-  Uncertainty
    * Examples that lie near the decision boundary of classifer model are picked.
-  Gaussian 
    * Gaussian function is used to pick examples based on their predictions and a gaussian curve. 


## Swipe Labeler:
[Swipe Labeler](https://github.com/spaceml-org/Swipe-Labeler) is a GUI based tool to enable labeling of data.
Supports:
- Multi-user labeling
- Compatibility with mobile devices

Images will be picked one by one from your unlabeled images directory, and presented through the Swipe Labeler GUI. For each image, the user can choose to classify the image as a positive or negative/absent class using the “Accept” or “Reject” button. 
For more info on how to install and use check [here](https://github.com/spaceml-org/Swipe-Labeler).

## Setup
Your data must be in the following format (inside a subdirectory) to be used by the pipeline:
```
Dataset/
└── Unlabeled
    ├── Image1.png
    ├── Image2.png
    └── Image3.png
```

## How to use?

* For Colab-friendly version and general examples check the [colab notebook](https://colab.research.google.com/drive/16SP5rPTuIaZeNqKMPSxw3NcuizGSgnn8?usp=sharing).
* For a CLI-Version, simply do:
   
  - Clone the code.
   
      ```git clone https://github.com/spaceml-org/Active-Labeler.git```
   
  - Install requirements.

      ```pip install -r requirements.txt```
   
  - Run the application.
   
      ```python3 main.py --config_path {path_to_config.yml} --class_name {positive_class_name}```

## Mandatory Arguments

__Pipeline Config__: Arguments used by the ```main.py``` present in [```pipeline_config.yml```](https://github.com/spaceml-org/Active-Labeler/blob/main/pipeline_config.yml)

| Argument     | Description        | 
| ------------ | ------------- | 
|```model_type```|     Type of Self-Supervised Learning model used - SimCLR or SimSiam|
|```model_path```|     Path to the .ckpt file containing the model (.ckpt file is obtained by training with [Self-Supervised Learner](https|//github.com/spaceml-org/Self-Supervised-Learner)|
|```image_size```|     Size of input images|
|```embedding_size```| Size of model's output representations |
|```verbose```|        1 to prints all logs, 0 to not print logs|
|```seed```| Seed used to save the state of a random functions used in the main pipeline for reproducibility |
|```data_path```| Path to dataset|
|```metrics```| 1 for empirical metrics about the pipeline by using predictions on the test dataset, 0 for no data |
|```metric_csv_path```| Path to csv containing metrics data|
|```prob_csv_path```| Path to csv containing prediction probilities for each iteration|
|```test_path```| Path to test dataset|
|```ref_img_path```| Path to reference image used to curate seed dataset|
|```unlabled_path```| Path to unlabled images for swipe labeler|
|```labeled_path```| Path to labeled images for swipe labeler|
|```positive_path```| Path to positive images for swipe labeler|
|```negative_path```| Path to negative images for swipe labeler|
|```unsure_path```| Path to unsure images for swipe labeler|
|```swipelabel_batch_size```| Labeling batch size for swipe labeler|
|```swipe_dir```| Path to the swipe labeler code|
|```num_trees```| Effects the build time and the index size. A larger value will give more accurate results, but larger indexes. More information on what trees in Annoy do can be found [here](https|//github.com/spotify/annoy#how-does-it-work).|
|```annoy_path```| Path to save .ann annoy file|
|```config_path```| Path to model_config.yml file|
|```sample_size```| Numbebr of images to be picked by the Active Labeler strategy|
|```sampling_strategy```| Active Labeler Strategy, Values| random, guassian, uncertainty |
|```batch_size```| Batch size for Active Labeler|
|```newly_labled_path```|  Path to store the labled images in the ongoing iteration|
|```archive_path```| Path to store the the labled images from the previous iterations|
|```nn```| 1 to do nearest neighbour search on the images picked by the Active Labeler strategy|
|```n_closest```| Number of nearest neighbour images for each strategy image|
|```al_folder```| Path to store your checkpoints and other files generated by the Active Labeler|
|```train_dataset_batch_size ```| Batch size for training dataset|
|```nn```| 1 to use nearest neighbour method on the reference image to curate seed dataset, 0 to use already existing seed_dataset|
|```seed_data_path```| Path to your existing seed_dataset

<br>

__Model Config__: Arguments related to model training present in [```model_config.yml```](https://github.com/spaceml-org/Active-Labeler/blob/main/model_config.yml)

| Argument     | Description        | 
| ------------ | ------------- | 
|```encoder_type```| Type of Self-Supervised Learning model used - SimCLR or SimSiam|
|```encoder_path```| Path to Self-Supervised Learning model|
|```e_embedding_size```| Size of encoder's output representations  |
|```e_lr```| Learning rate for encoder|
|```train_encoder```| True for training encoder, False for freezing the encoder during training|
|```classifier_type```| Architecture for classifier model, default| SSLEvaluator|
|```c_num_classes```| Number of classes to be classified, default| 2|
|```c_hidden_dim```| Dimension size of classfier model's hidden dim |
|```c_linear_lr```| Learning rate for classfier|
|```c_dropout```| Dropout rate for classifier|
|```c_gamma```| Gamma value for classifier|
|```c_decay_epochs```| Number of decay epochs for classifer |
|```c_weight_decay```| Weight decay for classifer|
|```c_final_lr```| Final learning rate for classifier|
|```c_momentum```| Momentum for classifier|
|```c_scheduler_type```| Type fo schedular used during training, values| cosine, step|
|```seed```| Seed used to save the state of a random functions used in model training for reproducibility|
|```cpus```| Number of cpus available for training|
|```device```| cuda if GPU is available, cpu otherwise|
|```epochs```| Number of training epochs


### Where can I find the trained model?
If needed, the finetuned model can be accessed from ```./final_model.ckpt```.

## Citation
If you find the Active Labeler useful in your research, please consider citing the github code for this tool:
```
@code{
  title={Active Labeler
},
author={Muthukrishnan, Subhiksha and Khokhar, Mandeep and Krishnan, Ajay and Narayanan, Tarun and Praveen, Satyarth and Koul, Anirudh}
  url={https://github.com/spaceml-org/Active-Labeler},
  year={2021}
}
```

</div>
