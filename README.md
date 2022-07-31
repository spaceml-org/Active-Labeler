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


## How to use?
  
  * The tool is meant to work on ipython environments to make proper use of the labeller. Refer to <notebook> as a template. 
  
## Configs
  
The pipeline is entirely controlled by yaml config files. 
There are 4 main keys

| Argument     | Description        | 
| ------------ | ------------- | 
|```model```| All model related args|
|```data```| All data related args |
|```train```| Model training related args|
|```active_learner```| Active Learning related args|
 
```model```
| Argument     | Description        | 
| ------------ | ------------- | 
|```model```| type of model to be loaded from ```./models/```|
|```model_path```| Path to the pretrained model weights|
|```device```| cuda/cpu to load the models and data on|
|```num_classes```| Number of classes the data should contain|
|```ssl```| parent key under model to enable the use of SSL models from Self-Supervised Learner|

```ssl```
| Argument     | Description        | 
| ------------ | ------------- | 
|```encoder```| ```encoder_type```: Type of encoder (SIMCLR,SIMSIAM) <br> ```encoder_path```: Path to load the encoder weights <br> ```train_encoder```:Freezes weights if not ```True``` <br> ```e_embedding_size```:Size of encoder's final layer output. <br> ```e_lr```:Learning Rate to be used by the encoder|
|```classifier```| ```classifier_type```: SSLEvaluator (Multiple FCs)/SSLEvaluatorOneLayer (Single FC) Classifer <br> ```c_num_classes```:Number of classes the data should contain <br> ```c_dropout```:Extent of dropout to be added for the fc layers <br> ```c_hidden_dim```:Hidden Dimension Size <br> ```c_lr```:Learning rate to be used by the classifier|
|```model_path```| Path to the pretrained model weights|


  
```data```
| Argument     | Description        | 
| ------------ | ------------- | 
|```dataset```| type of dataset (csv/tfds)|
|```path```| Path to the csv or tfds file containing data|
|```classes```| List of all classes present in the dataset|
  
```train```
| Argument     | Description        | 
| ------------ | ------------- | 
|```optimizer```| Optimizer to be used in the training loop. Specify the name under ```name``` key within this (Eg.```name```:SGD). Specify all related params under ```config```|
|```loss_fn```| Loss Funtion to be used in the training loop. Specify the name under ```name``` key within this (Eg.```name```:CrossEntropyLoss)|
|```batch_size```| Batch size per iteration|
|```epochs```| Number of epochs per iteration|

```active_learner```
| Argument     | Description        | 
| ------------ | ------------- | 
|```iterations```| Number of active learning iterations.|
|```strategy```| Uncertainty sampling technique. (Pick between ```margin_based```,```least_confidence```,```entropy_based```,```random_sampling```)|
|```num_labelled```| Number of samples to be labelled per iteration|
|```diversity_sampling```| Type of diversity sampling strategy to pick samples with uncertainty sampling. (Pick between ```pick_top_n```,```iterative_proximity_sampling```,```clustering_sampling```,```random_sampling```)|
|```limit```| -1|
|```preindex```| Enable preindex (True/False). Preindex - Use FAISS to retreive other similar examples based on every labelled data point.|
|```test_dataset```| Path to test dataset csv.|

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

  
