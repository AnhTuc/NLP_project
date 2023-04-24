#From tokenizer to models
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences =["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences)
outputs = model(**tokens)