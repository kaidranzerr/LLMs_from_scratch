# in GPT-2(124M parameters) the transformer block was repeated 12 times 
# expansion space is done so we can explore a richer space of parameters

# when a transformer block processes an input sequence each element is represented by a fixed size vector

# the operations within transformer block such as multi head attention and feed forward layer are designed to transform the input vectors
# such that dimensionality is preserved

# self attention analyzes relationship between input elements
# feed forward network --> modifies data individually at each position