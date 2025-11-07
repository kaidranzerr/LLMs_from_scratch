# subword based tokenization
# rule 1 . do not split frequently used words into smaller subwords
# rule 2: split the rare words into smaller , meaningful subwords

# the subword splitting helps the model learn that different words with same root word are similar in meaning
# it also helps the model learn that tokenization and modernization are made up of different root words

# BYTE PAIR ENCODING -->> subword tokenization algo || most common pair of consecutive bytes of data is replaced with a byte that 
# does not occur in data

# preprocessing --> add </w> at the end of each word
# then split words into characters and count their frequency
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = ''

integers = tokenizer.encode(text , allowed_special={"<|endoftext|>"})
 
enc_text = tokenizer.encode(text)
enc_sample = enc_text[50:]
# create input target pair
# it's an autoregresive model --> last output becomes current input
context_size = 4 # model is trained to look at sequence of 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

from torch.utils.data import Dataset , Dataloader 

class GPTDatasetV1(Dataset):
    def __init__(self , txt , tokenizer , max_length , stride):
        self.input_ids = []
        self.target_ids = []
        #tokenize the entire data
        token_ids = tokenizer.encode(txt , allowed_special=("<|endoftext|>"))

        # using a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0 , len(token_ids) - max_length , stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self , idx):
        return self.input_ids[idx] , self.target_ids[idx]

# drop_last = True drops the last batch if it is shorter than the specified batch size to prevent loss spikes during training
def create_dataloader_v1(txt , batch_size=4 , max_length=256 , stride=128 , shuffle=True , drop_last = True , num_workers = 0):
    # initialize the tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt , tokenizer , max_length , stride)
    dataloader = Dataloader(
        dataset , 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )