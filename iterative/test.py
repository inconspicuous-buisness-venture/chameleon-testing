from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    return set(lemma.name() for syn in synonyms for lemma in syn.lemmas())

def replace_with_synonym(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            new_word = synonyms.pop()  # Choose a synonym
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Example usage
text = "The quick brown fox jumps over the lazy dog."
modified_text = replace_with_synonym(text)
print(modified_text)



