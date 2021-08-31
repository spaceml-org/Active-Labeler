# Active Labeler
{image}

"When the neural network in your head has been trained to label images, but not to code"
The Active Labeller is a CLI Tool that facilitates labeling datasets with just a SINGLE line of code. The tool comes fully equipped with various Active Learning strategies and other customizable features to suit one’s requirements.

## (Not Necessary) Jargon
### What is Active Learning?
Bringing about a model’s full potential with limited data is the core concept behind Active Learning. This follows several strategies and other techniques that identify points based on a model’s difficulty in understanding them. 
Some of the strategies present in this tool includes:
-  Random
    * {}
-  Uncertainty
    * {}
-  Gaussian 
    * {}

## Swipe Labeller: (hyper linked to that page)
Swipe Labeller is a GUI based tool to enable labelling of data.
Supports:
- Multi-user labelling
- Compatibility with Mobile devices

Images will be picked one by one from your unlabeled images directory, and presented through the Swipe Labeler GUI. For each image, the user can choose to classify the image as a positive or negative/absent class using the “Accept” or “Reject” button. 
For more info on how to install and use [check](https://github.com/spaceml-org/Swipe-Labeler)

## Installation
{}

## How to use

* For Colab-friendly version and general examples check [colab]()
* For a CLI-Version, simply do 
    ```py
    python3 main.py --pipelineconfig path_to_config.json
    ```
__Mandatory Arguments__

```--model```: 

__Optional Arguments__
```eg```:

Your data must be in the following format to be used by the pipeline:
```
/Dataset
    Image1.png
    Image2.png
    Image3.png
```

### Where can I find the trained model?
If needed, the finetuned model can be accessed from ```path to model```

## Citation
If you find the Active Labeler useful in your research, please consider citing the github code for this tool:
```
@code{
  title={Active Labeller
},
  url={https://github.com/spaceml-org/Active-Labeller},
  year={2021}
}
```

</div>
