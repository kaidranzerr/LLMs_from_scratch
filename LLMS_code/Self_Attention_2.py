# self attention with trainable weights --> also called scaled dot product attention

# 1. we wanna compute context vectors as weighted sums over the input vectors specific to a certain input element
# 2. we will introduce weight matrices that are updated during model training
# 3. these trainable weight matrices are crucial so that model produces good context vectors
# we will implement self attention mechanism step by step by introducing 3 trainable weight matrices: query , key , value 
# these 3 matirces are used to project the embedded input tokens into query , key ,value vectors

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in , d_out) , requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in , d_out) , requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in , d_out) , requires_grad=False)

query_2 = x_2 @ W_query 
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

keys = inputs @ W_key 
values = inputs @ W_value 
queries = inputs @ W_query 


attn_scores_2 = query_2 @ keys.T # attention matrix between second query and all other keys 
attn_scores = queries @ keys.T 

# first query and all other keys 
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5 , dim=-1)

# why divide by sqrt --> stability in learning since softmax function is sensitive to magnitude of it's inputs and we can see peaks 
# division by sqrt makes the variance of dot product stable
# increase in variance increases with dimension so division by sqrt keeps the variance close to 1  

context_vec_2 = attn_weights_2 @ values 

# Final function implementation 
import torch.nn as nn 
class SelfAttention_v1(nn.Module):
    def __init__(self , d_in , d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in , d_out))
        self.W_key = nn.Parameter(torch.rand(d_in , d_out))
        self.W_value = nn.Parameter(torch.rand(d_in , d_out))
    
    def forward(self , x):
        keys = x @ self.W_key 
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax( attn_scores / keys.shape[-1] ** 0.5 , dim = -1)
        context_vec = attn_weights @ values 
        return context_vec 