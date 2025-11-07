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
