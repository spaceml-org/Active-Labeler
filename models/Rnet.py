import torch
import torchvision 
import torch.nn as nn


class Rnet(nn.Module):
  def __init__(self, **model_kwargs):
    super(Rnet, self).__init__()
    self.enc= torchvision.models.resnet50(pretrained = True)
    self.enc.fc = nn.Sequential(
    nn.Linear(2048, 1, bias = True),
    nn.Sigmoid()
    )

  def forward(self, x):
    x = self.enc(x)
    return x