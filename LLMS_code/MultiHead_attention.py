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
    
class MultiHeadAttention(nn.Module):
    def __init__(self , d_in , d_out , context_length , dropout , num_heads , qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_key = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_value = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.out_proj = nn.Linear(d_out , d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask" , torch.triu(torch.ones(context_length , context_length) , diagonal=1))
    def forward(self , x):
        b , num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value

        # we implicitly split the matrix by adding a 'num_heads' dimension
        # unroll last dim: (b , num_tokens , d_out) -> (b , num_heads , num_tokens , head_dim)
        keys = keys.view(b, num_tokens , self.num_heads , self.head_dim)
        values = values.view(b, num_tokens , self.num_heads , self.head_dim)
        query = query.view(b, num_tokens , self.num_heads , self.head_dim)

        # Transpose 
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens , :num_tokens]
        attn_scores.masked_fill_(mask_bool , -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5 , dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1,2)
        return context_vec
        # formulas for attention score --> queries * keys.T
        # context_vector == attention weights * values