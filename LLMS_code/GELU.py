# 2 activation functions commonly implemented in LLMs GELU and SwiGLU 
# cumulative distribution function of Standard Gaussian Distribution
# since it's very complicate we use approximation of GELU activation function approximation used for training GPT-2
# GELU is smooth throughout and is differentiable whereas RELU is not 
# GELU allows for non-zero output for negatve values thus solving dead neuron problem

# The inputs are projected into a 4 times larger space via first linear layer
# the second layer shrinks again by a factor of 4
# expansion and contraction allows for a rich exploration space
from torch import nn 
import torch 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self , x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *(x+0.044715 * torch.pow(x,3))
        ))
