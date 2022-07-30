<div align="center">
  <img src="https://github.com/spaceml-org/Active-Labeler/blob/review-master/active-simple-header.jpg" >
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

The tool supports a range of strategies which help find the data points that would contribute most to the accuracy of the model, in other words, the most 'influential' by computing their uncertainty scores. These are:
  - Least Confidence Sampling:
    * The farther away from the maximum probability, the more uncertain a datapoint can be.
  - Margin-Based Sampling:
    * This method is similar to Least Confidence Sampling except here, uncertainty is calculated by the difference in probabilities of the most confident and second most confident class.
  - Entropy based Sampling:
    * Previously mentioned methods do not work in a multi-class setting as information about all classes are not available.  In a general sense, entropy can be considered a measure of disorder or impurity in the system. It is widely used as a measure of the uncertainty of a model. Higher entropy value indicates that the model is highly uncertain about class membership.
    
There is a possibility that the samples selected for training are very similar to each other. In such a scenario, intuitively, the model would only learn about a certain type of image in each iteration, rendering the process inefficient. The inclusion of diversifying sampling strategies will help fully utilise each iteration, ensuring that the model learns from a set of diverse samples as opposed to a homogeneous one. Currently, the tool supports:
  - Pick top 'n' strategy:
    * Acting as the baseline, in this strategy we sample the top n most uncertain elements.
  - Iterative proximity-based approach:
    * This technique combines distances and uncertainty scores. A centroid is first calculated based on the position of the embeddings. Then their distance from the centroid is calculated and transformed to a scale of ```[-0.3,0.3]```. This distance is then added with the uncertainty scores and the top n scores are taken every iteration. 
  - Cluster Based Approach
    * We perform k-means clustering on the embeddings extracted from the penlultimate layer of the neural network. Here, k is a hyperparameter that controls the number of clusters formed. In this technique, clusters with bigger sizes contribute more data points to be selected and labelled.
  - Random sampling
    * Randomly selects points irrespective of their uncertainty scores.


## Setup
  
  explain about tfds, image, csv methods
  
## How to use?
  
  * For a CLI-Version, simply do:
   
  - Clone the code.
   
      ```git clone https://github.com/spaceml-org/Active-Labeler.git```
   
  - Install requirements.

      ```pip install -r ./Active-Labeler/requirements.txt```
   
  - Run the application.
   
      ```python3 ./Active-Labeler/main.py  --config_path {path_to_config.yaml} ```
  
## Mandatory Arguments

__Pipeline Config__: Arguments used by the ```main.py``` present in [```pipeline_config.yaml```]
  
