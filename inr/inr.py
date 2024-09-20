import torch.nn as nn
import torch


# Define the INR model (simple ReLU MLP)
class INR(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=256, num_hidden_layers=4, output_dim=3):
        super(INR, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return torch.sigmoid(self.output_layer(x))
