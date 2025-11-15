# another LLM decoding strategy 
from torch import nn 
import torch 
torch.manual_seed(123)
next_token_id = torch.multinomial(probas , num_samples=1).item()

# we want to restrict the oppurtunity of next token to only a few tokens

top_K = 3
top_logits , top_pos = torch.topk(next_token_logits , top_k)
# only the top tokens have the oppurtunity to become the next tokens

# logits --> top-k --> logits / temp --> softmax --> sample from multinomial 

# merge temperature scaling and top-k sampling 