# they are also known as skip connection or residual connections || to solve problem of vanishing gradient 
# gradients become progressively small as they propagate backwards 
# shortcut connection create an alternative path for the gradient to flow by skipping one or more layers 
# this is acheived by adding an output of a layer to a latter layer 
# +1 term keeps the gradient flowing through network
from torch import nn 
import torch 
class ExampleDeepNeuralNetwrok(nn.Module):
    def __init__(self , layer_sizes , use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0] , layer_sizes[1] , GELU())),
            nn.Sequential(nn.Linear(layer_sizes[1] , layer_sizes[2] , GELU())),
            nn.Sequential(nn.Linear(layer_sizes[2] , layer_sizes[3] , GELU())),
            nn.Sequential(nn.Linear(layer_sizes[3] , layer_sizes[4] , GELU())),
            nn.Sequential(nn.Linear(layer_sizes[4] , layer_sizes[5] , GELU()))
        ])

    def forward(self , x):
        for layer in self.layers:
            # compute output of current layer
            layer_output = layer(x)
            # check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x 
    