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


query = inputs[1] # 2nd input token is the query
attn_scores_2 = torch.empty(inputs.shape[0])
for i,x_i in enumerate(shape):
    attn_scores_2[i] = torch.dot(x_i , query)
print(attn_scores_2)

# next step is normalization --> to obtain attention weights that sum up to 1
attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()
# using softmax is more advisable 
