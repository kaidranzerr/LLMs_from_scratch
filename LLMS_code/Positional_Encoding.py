# in embedding layer same token ID gets mapped to same vector representation regardless of where toekn ID is positioned in input sequence
# it is helpful to inject additional position information to LLM
# 2 types of positional embeddings (Absolute , Relative)

# Absolute --> for each position in input sequence , a unique embedding is added to token embedding to convey it's exact location
# positional vectors have same dimension as original token embeddings

# Relative --> The emphasis is on the relative position or distance between tokens .
# The model learns the relationships in terms of how far apart rather than at which exact position

# Absolute --> suitable when fixed order of token is crucial such as sequence generation
# Relative --> Suitable for tasks like language modelling over long sequence