import torch


class ClassifierModel(torch.nn.Module):
    """Doc string"""

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

    def forward(self, x):
        """Forward pass of the model"""
        if self.train_encoder:
            feats = self.encoder(x)[-1]
            feats = feats.view(feats.size(0), -1)
            logits = self.linear_model(feats)
        else:
            logits = self.linear_model(x)
        return logits
