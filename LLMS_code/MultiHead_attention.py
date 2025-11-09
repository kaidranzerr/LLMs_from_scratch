# multihead attention --> the term multi-head refers to dividing attention mechanism into multiple heads (each operating independently)
# stacking multiple single head attention layers
# 1.Implementing multi-head attention involves creating multiple instances of self-atttention mechanism each with it's own weights and
# then combining their outputs , this can be computationally intensive but it makes LLMs powerful at complex pattern recognition tasks

# extending single head attention to multi head attention