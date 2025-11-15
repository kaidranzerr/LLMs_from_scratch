# another LLM decoding strategy 
from torch import nn 
import torch 
torch.manual_seed(123)
next_token_id = torch.multinomial(probas , num_samples=1).item()

# we want to restrict the oppurtunity of next token to only a few tokens