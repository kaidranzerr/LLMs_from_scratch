# training deep NN can be challenging due to vanishing/exploding gradients
# this leads to unstable training dynamics
# as training proceeds inputs to each layer can change internal covariate shift --> This delays convergence --> Layer normalization prevents this
# If layer output is too large or small gradient magnitudes can become too large or small || this affects training 
# Layer normalization keeps gradient stable
import torch
import torch.nn as nn
# main idea --> adjust output of NN to have mean 0 and variance of 1 This speeds up convergence 
torch.manual_seed(123)
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6) , nn.ReLU())
out = layer(batch_example)
print(out)