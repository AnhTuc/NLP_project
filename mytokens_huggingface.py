#From hugging face
from transformers import AutoTokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
checkpoint2 = 'bert-base-uncased'
#Download vocabulary the same as the one model pretrained with
tokenizer = AutoTokenizer.from_pretrained(checkpoint2)
"""
How tokenizer works
"""
sequence = "Let's have a chat"
#This tokenizer is a subword tokenizer: it splits the words until it obtains tokens 
# that can be represented by its vocabulary.
tokens = tokenizer.tokenize(sequence)
print(tokens)

#get numberical representation of word from dictionary
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)

#Add special starting and ending sentence to tokens
final_input = tokenizer.prepare_for_model(input_ids)

"""
Short of tokenizer
"""
final_tokens = tokenizer('Let\'s have a chat')
print(final_tokens)
#Note that attention mask use to indicate which tokens are padding and should not be important
#Use attention mask with flag: tokenizer(sequences, padding = True)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
tokesn = tokenizer.tokenize("I have a new gpu")