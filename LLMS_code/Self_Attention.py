# Self-Attention
# encoder compress entire input sequence into a single hidden state vector
# in self attention the self refers to the mechanism ability to compute attention weights by relating different positions in a single
# input sequence
# it learns the relationships between various parts of input itself , such as words in sentence 

# simplified attention mechanism
# context vectors --> attention weights to weigh importance of each input

# implementing a simplified attention mechanism 
# taking 3D embedding vectors

# the element or token which we are looking at right now is also called query
# dot product is used to check alignment between the 2 vectors || higher the dot product higher the similarity and attention scores between 2 elements

import torch
query = inputs[1] # 2nd input token is the query
attn_scores_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(shape):
    attn_scores_2[i] = torch.dot(x_i , query)
print(attn_scores_2)

# next step is normalization --> to obtain attention weights that sum up to 1
attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()
# using softmax is more advisable 

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
attn_scores_2_naive = softmax_naive(attn_scores_2)

# pytorch implementation of softmax
attn_weights_2 = torch.softmax(attn_scores_2 , dim=0)
# after computing the normalized attention weights we calculate the context vectors by multiplying the embedded input tokens with
# corresponding attention weights and the summing the resultant vectors 

query = inputs[1]
context_vector_2 = torch.zeros(query.shape)
for i,x_ in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i 


attn_scores = torch.empty(6,6)
for i,x_i in enumerate(inputs):
    for j , x_j in enumerate(inputs):
        attn_scores[i , j] = torch.dot(x_i , x_j)
print(attn_scores) 

# matrix multi
attn_scores = inputs @ inputs.T 
print(attn_scores)

attn_weights = torch.softmax(attn_scores , dim = -1) # implementing softmax to each row

# computing the context vectors 
all_context_vectors = attn_weights @ inputs 