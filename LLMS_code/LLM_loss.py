# tokenizer converts tokens into tokenIds
# GPTModel converts tokenIDs into logits
# Logits are converted back into tokenIDs
from torch import nn 
import torch
# batch_size , num_tokens  , vocab dimensions
inputs = torch.tensor([[16833 , 3626 , 6100],
                       [40 , 1107 , 588]])

targets = torch.tensor([[3626 , 6100 , 345],
                        [1107 , 588 ,11311]])

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits , dim=-1)
print(probas.shape) 

token_ids = torch.argmax(probas , dim=-1 , keepdim=True)

# token probabilities corresponding to target indices
text_idx = 0
target_probas_1 = probas[text_idx , [0,1,2] , targets[text_idx]]

text_idx = 1
target_probas_2 = probas[text_idx , [0,1,2] , targets[text_idx]]

# compute logarithm and also concatenate
log_probas = torch.log(torch.cat((target_probas_1 , target_probas_2)))
print(log_probas)
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
neg_avg_log_probas = avg_log_probas * -1

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()

loss = torch.nn.functional.cross_entropy(logits_flat , targets_flat)
print(loss) 

# perplexity --> Another loss measure like cross entropy 
# measures how well the probability distribution predicted by model matches the actual distribution of words in dataset
# more interpretable way of understanding model uncertainity in predicting next token 
# lower perplexity score --> better predictions
perplexity = torch.exp(loss)

