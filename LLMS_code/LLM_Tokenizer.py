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

# vocabulary
