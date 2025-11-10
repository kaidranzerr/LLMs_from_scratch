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

mean = out.mean(dim=-1 , keepdim=True)
var = out.var(dim=-1 , keepdim=True)

out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1 , keepdim=True)
var = out_norm.var(dim=-1 , keepdim=True)

class LayerNorm(nn.Module):
    def __init__(self , emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self , x):
        mean = x.mean(dim=-1 , keepdim=True)
        var = x.var(dim=-1 , keepdim=True , unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var+self.eps)
        return self.scale * norm_x + self.shift 
    
# scale and shift are 2 trainable parameters which have same dimension as input parameter (finetuning parameters)
# if unbiased = True we will apply something called Bessels correction which typically divides by n-1 not n to adjust for bias in sample
# variance estimation