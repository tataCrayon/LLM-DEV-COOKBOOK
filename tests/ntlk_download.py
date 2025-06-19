import nltk
nltk.data.path.append(r'F:\DependencyPackages\llmRepository\nltk_data')
from nltk.tokenize import word_tokenize
text = "Hello, world!"
tokens = word_tokenize(text)
print(tokens)