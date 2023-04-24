import collections

class SubwordTokenizer(object):

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = collections.defaultdict(int)

    def fit(self, text):
        for word in text.split():
            self.vocab[word] += 1

    def tokenize(self, text):
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                for i in range(len(word)):
                    subword = word[:i] + "_" + word[i:]
                    if subword in self.vocab:
                        tokens.append(subword)
                        break
            else:
                tokens.append(word)
        return tokens
