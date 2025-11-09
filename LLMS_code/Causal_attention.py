# causal attention also known as masked attention is a special form of self attention
# it restricts the model to only consider previous and current inputs in a sequence when processing any given token
# this is contrast to any self attention mechanism which allows access to entire input sequence at once 
# the causal attention mechanism ensures that model only factors in tokens that occur at or before the current token in the sequence
# we mask out future tokens which comes after current token in input text 

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T 
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5 , dim=1)

# masking function
context_length = attn_scores.shape[0]
mask_sample = torch.tril(torch.ones(context_length , context_length)) # using the lower triangular matrix
print(mask_sample)


mask_simple = attn_weights * mask_sample
# normalize the attention weights
row_sums = masked_simple.sum(dim=1 , keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

# to prevent data leakage --> attention scores --> Upper Triangular Matrix --> Softmax 

mask = torch.triu(torch.ones(context_length , context_length) , diagonal = 1)
masked = attn_scores.masked_fill(mask.bool() , -torch.inf) # cancelled the influence of future tokens
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5 , dim = 1)
print(attn_weights)

# masking additional attention weights with dropout
# dropout is a deep learning technique where randomly selected hidden layer units are ignored during training
# improves overfitting and improves generalization performance 

# using dropout rate of 50% 
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))
print(dropout(attn_weights))

# complete algorithm implementation
batch = torch.stack((inputs , inputs) , dim = 0)
print(batch.shape)

class CausalAttention(nn.Module):
    def __init__(self , d_in , d_out , context_length , dropout , qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_key = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.W_value = nn.Linear(d_in , d_out , bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask' , torch.triu(torch.ones(context_length , context_length), diagonal = 1))

    def forward(self , x):
        b , num_tokens , d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill(self.mask.bool()[:num_tokens , :num_tokens] , -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5 , dim=-1)
        attn_weights = self.dropout(attn_weights) 
        context_vec = attn_weights @ values
        return context_vec 