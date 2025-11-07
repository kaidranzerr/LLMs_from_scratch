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

