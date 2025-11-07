import re

with open('the-verdict.txt' , "r" , encoding='utf-8') as f:
    raw_text = f.read()

text = "Hello World. This is a test"
result = re.split(r'([,.:;!&?_"()\']|\s)' , text)
print(result)

res = [item for item in result if item.strip()] #getting rid of whitespaces
print(res)

preprocessed = re.split(r'([,.:;!&?_"()\']|\s)' , raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])

# vocabulary --> map these tokens in alphabetical order to unique ids
# each unique token is mapped to an unique integer called token id

# create token ids
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i ,item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# inverse version of vocab is also necessary that maps token_ids back to original text
class SimpleTokenizerV1:
    def __init__(self , vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    def encode(self , text):
        preprocessed = re.split(r'([,.:;!&?_"()\']|\s)' , text) 
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]       
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids 
    def decode(self , ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # replace spaces before specified punctuations
        text = re.sub(r'\s+([,.!?"()\])' , 'r\1' , text)
        return text 