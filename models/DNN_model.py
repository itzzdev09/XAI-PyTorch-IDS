import torch
import torch.nn as nn

class DNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3):
        super(DNNModel, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
