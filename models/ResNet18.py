import torch
import torchvision
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self, **model_kwargs):
        super(ResNet18, self).__init__()
        self.enc = torchvision.models.resnet18(pretrained=True)
        self.enc.fc = nn.Identity()
        self.true_fc = nn.Linear(512, 21)

    def forward(self, x, want_embeddings=False):

        embedding = self.enc(x)

        out = self.true_fc(embedding)

        if want_embeddings == True:
            return out, embedding
        else:
            return out
