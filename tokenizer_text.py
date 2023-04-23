import tensorflow as tf

def tokenize(text):
  # Tokenize the text using the BERT tokenizer.
  tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<unk>')
  tokenizer.fit_on_texts([text])
  tokens = tokenizer.texts_to_sequences([text])

  # Convert the tokens to subwords.
  subwords = []
  for token in tokens:
    subwords.extend(tokenizer.word_to_subwords(token))

  return subwords

def embed(subwords):
  # Create a BERT embedding layer.
  embedding_layer = tf.keras.layers.Embedding(
      len(tokenizer.word_index) + 1,
      768,
      trainable=False,
      mask_zero=True)

  # Embed the subwords.
  embeddings = embedding_layer(tf.keras.layers.Input(shape=(None,)))

  return embeddings