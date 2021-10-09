<div align="center">
  <img src="https://github.com/spaceml-org/Active-Labeler/blob/main/active-simple-header.jpg" >
<p align="center">
  Published by <a href="http://spaceml.org/">SpaceML</a> •
  <a href="https://arxiv.org/abs/2012.10610">About SpaceML</a>
</p>

[![Python Version](https://img.shields.io/badge/python-3.5%20|%203.6%20|%203.7%20|%203.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/Cuda-10%20|%2011.0-4dc71f.svg)](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html)
[![Pip Package](https://img.shields.io/badge/Pip%20Package-Coming%20Soon-0073b7.svg)](https://pypi.org/project/pip/)
[![Docker](https://img.shields.io/badge/Docker%20Image-Coming%20Soon-34a0ef.svg)](https://www.docker.com/)
   
[![Google Colab Notebook Example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18FDsvNsTm65sshmo9XYU04ITqmfgSEv4?usp=sharing)
   
</div>

The Active Labeler is a CLI tool that facilitates labeling datasets with just a SINGLE line of code. The tool comes fully equipped with various Active Learning strategies and other customizable features to suit one’s requirements.

## What is Active Learning?
Deep learning has a strong greedy attribute to the labeled data. While, in the real world, obtaining a comprehensive set of unlabeled datasets is relatively simple, the manual labeling of data comes at a high cost; this is especially true for those fields where labeling requires a high degree of expert insight. A way of maximizing the performance gain of a deep learning model while labeling a small number of images can significantly impact the practical implementations of AI in multiple fields. Active learning is such a method. It aims to select the most valuable samples from the unlabeled dataset and transfer them to a human annotator for labeling. This method of selective sampling of data to label reduces the cost of labeling while still maintaining performance.
Some of the strategies present in this tool includes:
-  [Random](https://numpy.org/doc/stable/reference/random/index.html)

    * Random function is used to pick examples uniformly from the entire unlabeled dataset. 
-  [Uncertainty](https://livebook.manning.com/book/human-in-the-loop-machine-learning/chapter-3/172)
    * Examples that lie near the decision boundary of classifer model are picked.
-  Gaussian 
    * Gaussian function is used to pick examples based on their predictions and a gaussian curve. 

<br>
<img src="https://github.com/spaceml-org/Active-Labeler/blob/main/ActiveLabeler_diagram.jpeg" >

## Swipe Labeler:
[Swipe Labeler](https://github.com/spaceml-org/Swipe-Labeler) is a GUI based tool to enable labeling of data.
It supports:
- Multi-user labeling
- Compatibility with mobile devices

Images will be picked one by one from your unlabeled images directory, and presented through the Swipe Labeler GUI. For each image, the user can choose to classify the image as a positive or negative/absent class using the “Accept” or “Reject” button. 
For more info on how to install and use check [here](https://github.com/spaceml-org/Swipe-Labeler).

When running on Colab, Swipe Labeler cannot be accessed at ```https://localhost:5000/```. Run the following code to obtain an url that allows you to access the Swipe Labeler.
<br>```from google.colab.output import eval_js```
<br>```print(eval_js('google.colab.kernel.proxyPort(5000)'))  ```

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

* For Colab-friendly version and general examples check the [colab notebook](https://github.com/spaceml-org/Curator-Unlabeled-Image-Search-Guide/blob/main/notebooks/SSL%2BImage_Similarity_Search%2BSwipe_Labeler%2BActive_Labeler.ipynb).
* For a CLI-Version, simply do:
   
  - Clone the code.
   
      ```git clone https://github.com/spaceml-org/Active-Labeler.git```
   
  - Install requirements.

      ```pip install -r ./Active-Labeler/requirements.txt```
   
  - Run the application.
   
      ```python3 ./Active-Labeler/main.py  --config_path {path_to_config.yaml} ```

## Mandatory Arguments

__Pipeline Config__: Arguments used by the ```main.py``` present in [```pipeline_config.yaml```](https://github.com/spaceml-org/Active-Labeler/blob/main/pipeline_config.yaml)

| Argument     | Description        | 
| ------------ | ------------- | 
|```verbose```|        ```1``` prints all logs, ```0``` does not print logs|
|```seed```| Seed used to save the state of a random functions used in the main pipeline for reproducibility |
|```data_path```| Path to dataset|
|```runtime_path```| Path to folder where all runtime files will be stored|
|```swipe_label_batch_size```| Labeling batch size for Swipe Labeler|
|```model_type```|     Type of Self-Supervised Learning model used: ```SimCLR```, ```SimSiam```|
|```model_path```|     Path to the .ckpt file containing the model (.ckpt file is obtained by training with [Self-Supervised Learner](https://github.com/spaceml-org/Self-Supervised-Learner))|
|```image_size```|     Size of input images|
|```embedding_size```| Size of model's output representations |
|```model_config_path```| Path to ```model_config.yaml``` file|
|```seed_nn```| ```1``` to use nearest neighbour method on the reference image to curate seed dataset, ```0``` to use already existing seed_dataset|
|```ref_img_path```| Path to reference image used to curate seed dataset, needed if ```seed_nn``` is  ```1```|
|```seed_data_path```| Path to your existing seed_dataset, needed if ```seed_nn``` is  ```0```|
|```num_trees```| Effects the annoy tree build time and the index size. A larger value will give more accurate results, but larger indexes. More information on Annoy trees can be found [here](https://github.com/spotify/annoy#how-does-it-work).|
|```sample_size```| Numbebr of images to be picked by the Active Labeler strategy|
|```sampling_strategy```| Type of Active Labeler strategy used: ```random```, ```guassian```, ```uncertainty``` |
|```active_label_batch_size```| Batch size for Active Labeler|
|```sampling_nn```| ```1``` to do nearest neighbour search on the images picked by the Active Labeler strategy in each iteration|
|```n_closest```| Number of nearest neighbour images for each strategy image|
|```train_dataset_batch_size ```| Batch size for training dataset|
|```metrics```| ```1``` to obtain empirical metrics about the pipeline by using predictions on the validation dataset, ```0``` for no data |
|```pos_class```| Name of positive class, needed if ```metrics``` is ```1``` or if ```simulate_label``` is ```1```|
|```metric_csv_path```| Path to csv containing metrics data|
|```prob_csv_path```| Path to csv containing prediction probilities for each iteration|
|```simulate_label```| Function that simulates labeling for testing purposes by check if ```pos_class``` is part of the image name |

<br>

__Model Config__: Arguments related to model training present in [```model_config.yaml```](https://github.com/spaceml-org/Active-Labeler/blob/main/model_config.yaml)

| Argument     | Description        | 
| ------------ | ------------- | 
|```encoder_type```| Type of Self-Supervised Learning model used: ```SimCLR``` or ```SimSiam```|
|```encoder_path```| Path to Self-Supervised Learning model|
|```e_embedding_size```| Size of encoder's output representations  |
|```e_lr```| Learning rate for encoder|
|```train_encoder```| ```True``` for training encoder, ```False``` for freezing the encoder during training|
|```classifier_type```| Architecture for classifier model: ```SSLEvaluator``` for multiple layers, ```SSLEvaluatorOneLayer``` for single layer|
|```c_num_classes```| Number of classes to be classified|
|```c_hidden_dim```| Dimension size of classfier model's hidden dim |
|```c_linear_lr```| Learning rate for classfier|
|```c_dropout```| Dropout rate for classifier|
|```c_gamma```| Gamma value for classifier|
|```c_decay_epochs```| Number of decay epochs for classifer |
|```c_weight_decay```| Weight decay for classifer|
|```c_final_lr```| Final learning rate for classifier|
|```c_momentum```| Momentum for classifier|
|```c_scheduler_type```| Type of schedular used during training: ```cosine```, ```step```|
|```seed```| Seed used to save the state of a random functions used in model training for reproducibility|
|```cpus```| Number of cpus available for training|
|```device```| ```cuda``` if GPU is available, ```cpu``` otherwise|
|```epochs```| Number of training epochs|


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
