import torch
import torchvision
import torch.nn as nn


class Rnet(nn.Module):
    def __init__(self, **model_kwargs):
        super(Rnet, self).__init__()
        self.enc = torchvision.models.resnet50(pretrained=True)
        self.enc.fc = nn.Identity()
        self.true_fc = nn.Sequential(nn.Linear(2048, model_kwargs['num_classes'], bias=True))

    def forward(self, x, want_embeddings=False):
        embedding = self.enc(x)

        out = self.true_fc(embedding)

        if want_embeddings == True:
            return out, embedding
        else:
            # return [batch_size] shape instead of [batch_size,1]
            return out