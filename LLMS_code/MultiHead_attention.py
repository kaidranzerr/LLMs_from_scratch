# multihead attention --> the term multi-head refers to dividing attention mechanism into multiple heads (each operating independently)
# stacking multiple single head attention layers
# 1.Implementing multi-head attention involves creating multiple instances of self-atttention mechanism each with it's own weights and
# then combining their outputs , this can be computationally intensive but it makes LLMs powerful at complex pattern recognition tasks

# extending single head attention to multi head attention 
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self , d_in , d_out , context_length , dropout , num_heads , qkv_bias=False):
        super.__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in , d_out , context_length , dropout , qkv_bias)
                                    for _ in range(num_heads)])
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads] , dim=-1)