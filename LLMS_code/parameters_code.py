from torch import nn 
import torch
class GPTModel(nn.Module):
    def __init__(self , cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"] , cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"] , cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"] , cfg["vocab_size"] , bias=False
        )

    def forward(self , in_idx):
        batch_size , seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len , device=in_idx.device))
        x = tok_embeds + pos_embeds # shape [batch_size , num_tokens , emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits 
    
total_params = sum(p.numel() for p in model.parameters())
# .numel() returns total number of elements in given tensor
# generate text from output tensor

def generate_text_simple(model , idx , max_new_tokens , context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[: , -context_size:] # restricts the input so we only look at tokens == context size
    # get the predictions
        with torch.no_grad():
            logits = model(idx_cond)  ## batch , n_tokens , vocab_size
        logits = logits[: , -1 , :] # extracting the last row from the logits tensor
        probas = torch.softmax(logits , dim =-1)  # softmax applied to each row
        idx_next = torch.argmax(probas , dim=-1 , keepdim=True)  # index with highest probability
        idx = torch.cat((idx , idx_next) , dim=1) #(batch , n_tokens+1) appending part is done for the next round
    return idx

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)