# in GPT-2(124M parameters) the transformer block was repeated 12 times 
# expansion space is done so we can explore a richer space of parameters

# when a transformer block processes an input sequence each element is represented by a fixed size vector

# the operations within transformer block such as multi head attention and feed forward layer are designed to transform the input vectors
# such that dimensionality is preserved

# self attention analyzes relationship between input elements
# feed forward network --> modifies data individually at each position
import torch
from torch import nn
GPT_CONFIG_124M = {
    "vocab_size":50257,
    "context_length":1024,
    "emb_dim":768,
    "n_heads":12,
    "n_layers":12,
    "drop_rate":0.1,
    "qkv_bias":False
}

class TransformerBlock(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout = cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self , x):
        # shortcut connection for attention block
        shortcut = x 
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut 

        # shortcut connection for feed forward block
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x