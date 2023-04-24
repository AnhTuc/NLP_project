import collections
import itertools
import random
train_data = "This is an example. Marie goes to school. She does homework."
threshold = 2

#Create a vocabulary
vocab = set()
for word in train_data.split():
    vocab.add(word)

#Create a list of pairs of characters.
pairs = []
for i in range(len(train_data) - 1):
    pairs.append((train_data[i], train_data[i + 1]))

#Create a list of frequencies for each pair of characters.
frequencies = collections.Counter(pairs)

#Sort the pairs by frequency.
sorted_pairs = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

#Start with an empty set of subwords.
subwords = set()
#Iterate over sorted pairs
for pair, frequency in sorted_pairs:

    # If the pair is not already in the vocabulary, add it.
    if pair not in vocab:
        vocab.add(pair)

    # If the frequency of the pair is greater than a threshold, add it to the set of subwords.
    if frequency > threshold:
        subwords.add(pair)

def tokenize(text):
    tokens = []
    for word in text.split():
        for subword in subwords:
            if subword in word:
                tokens.append(subword)
                break
        else:
            tokens.append(word)
    return tokens

print(tokenize("This is a test slang"))