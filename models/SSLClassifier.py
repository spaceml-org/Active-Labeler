import torch
from torch import nn


class ClassifierModel(torch.nn.Module):
    def __init__(self, device, encoder, linear_model):
        """
        Loads the encoder and the linear_model to gpu or cpu
        depending on the device.

        Keyword arguments
        device -- Hardware used for running the pipeline. cuda or cpu
        encoder -- Loaded encoder from the SSL repo
        linear_model -- Initialised classification head for the encoder
        """
        super().__init__()
        # Models
        self.encoder = encoder
        if self.encoder.fc:
            self.encoder.fc = nn.Identity()
        self.linear_model = linear_model
        self.train_encoder = True

        self.encoder.to(device)
        self.linear_model.to(device)

    def freeze_encoder(self):
        """Freezes the encoder"""
        self.train_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreezes the encoder"""
        self.train_encoder = True
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x, want_embeddings=False):
        """Forward pass of the model"""
        feats = self.encoder(x)[-1]
        feats = feats.view(feats.size(0), -1)
        logits = self.linear_model(feats)
        if want_embeddings == True:
            return logits, feats
        else:
            return logits

    def is_frozen(self):
        return self.train_encoder
