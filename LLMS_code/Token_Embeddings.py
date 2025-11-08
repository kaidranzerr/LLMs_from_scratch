# representing words numerically
# semantically similar words should have similar vectors
# we gonna be needing token IDs | Vector Dimension | Vocab size

# creating token embeddings
input_ids = torch.tensor([2,3,5,6])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size , output_dim) # initialize the vector embeddings || initialize weight of embedding matrix
# in a random manner || it is a lookup table
print(embedding_layer.weight)
# embedding layer is a look up operation that retrieves rows from the embedding layer weight matrix using a token ID.

# how token embedding are created for LLMs
# 1. Initialize embedding weights with random values
# 2. Initialization serves as starting point for LLM training process
# 3. The embedding weights are optimized as part of LLM trianing process
print(embedding_layer(torch.tensor[3]))