# temperature scaling --> replace argmax with probability distribution (multinomial prob distribution samples next token according 
# probability score)
# top k sampling 

# Temperature scaling controls an LLM's output by adjusting its randomness; a low temperature makes outputs more predictable and focused,
#  while a high temperature encourages more creative, diverse, and surprising results.
#  It works by scaling the probabilities of the next words the model considers

# temperature is just a fancy term for dividing the logits by a number greater than 0
# scaled logits = logits / temperature
# small --> sharper distribution || large --> flattened distribution
import torch 
from torch import nn 
def softmax_with_temperature(logits , temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits , dim = 0)
temperatures = [1 , 0.1 , 5]

scaled_probas = [softmax_with_temperature(next_token_logits , T) for T in temperatures]
