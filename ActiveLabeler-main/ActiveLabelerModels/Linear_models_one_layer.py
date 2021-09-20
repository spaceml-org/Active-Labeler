from torch import nn


class SSLEvaluatorOneLayer(nn.Module):
    """The classification model which is used with the SSL encoder"""

    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1, c_type="Sigmoid"):
        super().__init__()
        """
        Keyword arguments
        n_input -- Size of the input embeddings
        n_classes -- Number of output classes (default 1 for Sigmoid)
        n_hidden -- Input reduction dims
        p -- Dropout input
        c_type -- Classification type: Sigmoid or Softmax 
        """
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        if c_type == "Sigmoid":
            if n_hidden is None:
                # use linear classifier
                self.block_forward = nn.Sequential(
                    Flatten(),
                    nn.Dropout(p=p),
                    nn.Linear(n_input, 1, bias=True),
                    nn.Sigmoid(),
                )
            else:
                # use simple MLP classifier
                self.block_forward = nn.Sequential(
                    Flatten(),
                    nn.Dropout(p=p),
                    nn.Linear(n_input, n_hidden, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_hidden, 1, bias=True),
                    nn.Sigmoid(),
                )
        else:
            if n_hidden is None:
                # use linear classifier
                self.block_forward = nn.Sequential(
                    Flatten(),
                    nn.Dropout(p=p),
                    nn.Linear(n_input, n_classes, bias=True),
                    nn.Softmax(),
                )
            else:
                # use simple MLP classifier
                self.block_forward = nn.Sequential(
                    Flatten(),
                    nn.Dropout(p=p),
                    nn.Linear(n_input, n_hidden, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(n_hidden, n_classes, bias=True),
                    nn.Softmax(),
                )

    def forward(self, x):
        """Forward pass"""
        logits = self.block_forward(x)
        return logits


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input_tensor):
        """Forward pass"""
        return input_tensor.view(input_tensor.size(0), -1)
