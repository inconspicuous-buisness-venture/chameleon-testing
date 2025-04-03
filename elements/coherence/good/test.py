import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")  # Adds TextRank for entity-based coherence

text = "John loves football. He plays it every weekend with his friends."
doc = nlp(text)

for phrase in doc._.phrases:
    print(phrase.text, phrase.rank)  # Higher rank = more coherence