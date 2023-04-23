import re

def tokenize(word):
  """
  Tokenizes a word into its bare form and suffix.

  Args:
    word: The word to tokenize.

  Returns:
    A tuple of the bare form and the suffix.
  """

  # Split the word into its stem and suffix.
  stem, suffix = re.split(r'(?<=[aeiou])(s|ed|ing|es|ly|er|est)', word)

  # Return the tuple of the stem and the suffix.
  return stem, suffix