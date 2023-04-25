corpus =[
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

#Load for normalization and pre-tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#Caculate word's frequency
from collections import defaultdict
word_freqs = defaultdict(int)

#Word dictionary
for text in corpus:
    word_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    print(word_with_offsets)
    new_words = [word for word, offset in word_with_offsets]
    for word in new_words:
        word_freqs[word]+=1

#Create base vocabulary
#Alphabet
alphabet = []
for word in word_freqs:
    if word[0] not in alphabet:
        alphabet.append(word[0])
    for letter in word:
        if f"##{letter}" not in alphabet:
            alphabet.append(f"##{letter}")

alphabet.sort()
#Base vocab
vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

#Split all words in corpus into first and ##letter
splits = {word:[c if i == 0 else f"##{c}" for c,i in enumerate(word)]
          for word in word_freqs.keys()}

#score function for each pair
def compute_pair_scores(splits):
    """
    Compute a pair score to choose highest score - pair to merge
    """
    letter_freqs, pair_freqs = defaultdict(int), defaultdict(int)
    for word, freqs in word_freqs.items():
        split_word = splits[word]
        if len(split_word) ==1:
            letter_freqs+=freqs
            continue
        
        for i in range(len(split_word)-1):
            pair_words = (split_word[i], split_word[i+1])

            letter_freqs[split_word[i]] +=freqs
            pair_freqs[pair_words] += freqs
        letter_freqs[split_word[-1]] +=freqs
    
    score = {
        letters : freqs/(letter_freqs[letters[0]] + letter_freqs[letters[1]]) 
        for letters, freqs in pair_freqs.items()
    }
    return score


#Merge higher score pair
def merge_pair(a,b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split)-1):
            if split[i] == a and split[i+1] == b:
                merge = a + b[2:]
                split = split[:i] +[merge] +split[i+2:]
        splits[word] = split
    return splits

def wordpiece(vocab, desired_vocab_len = 70):
    """
    Loop to find best pair to add to vocab until it reaches desired length
    """
    while len(vocab) <desired_vocab_len:
        scores = compute_pair_scores(splits)
        best_pair, max_score = "", None

        for pair, score in scores:
            if max_score ==None or max_score<score:
                best_pair = pair
                max_score = score

        splits = merge_pair(*best_pair, splits)

        new_tokens = best_pair[0]
        new_tokens+= best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[1]

        vocab.append(new_tokens)


def encode_word(word):
    """
    Encode a given word
    """
    tokens =[]
    while(len(word) >0):
        i = len(word)
        while i>0 and word[:i] not in vocab:
            i-=1
        if i == 0:
            return "[UNK]"
        tokens.append(word[:i])
        word = word[i:]

        if len(word) >=0 :
            word = f"##{word}"
    return tokens

def decode(tokens, vocab):
    """
    Decode a given tokens
    """
    seq =""
    for token,i in enumerate(tokens):
        if token.startswith("##"):
            new_token = token[2:]
        elif i!=0:
            new_token = ' '+ token
        else:
            new_token = token
        seq += new_token

def tokenize(text):
    pre_tokenize_text = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenize_word = [word for word, offset in pre_tokenize_text]
    encode_words = [encode_word(word) for word in pre_tokenize_word]
    return encode_words

if __name__ =="__main__":
    wordpiece(vocab)
    tokenize("Lan does homework")



