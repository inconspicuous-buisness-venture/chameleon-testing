import re
import string
import spacy
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np


# Load spaCy English model (make sure to run: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Predefined stopwords (this could be extended)
STOPWORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}

# ---------------------------
# Heuristic Filter Functions
# ---------------------------

def has_first_letter_caps(text):
    """Check if the first character is uppercase."""
    text = text.strip()
    return bool(text) and text[0].isupper()

def no_all_caps(text):
    """Return True if text is not all uppercase."""
    return not text.isupper()

def word_repetition_ratio_ge_0_2(text):
    """Check if the ratio of repeated words exceeds 0.2.
       Here, we consider repetition if any word appears more than once.
    """
    words = re.findall(r'\w+', text.lower())
    if not words:
        return False
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1
    # Count how many words are repeated
    repeated = sum(1 for count in word_counts.values() if count > 1)
    return (repeated / len(words)) > 0.2

def digit_punctuation_ratio(text):
    """Check if the ratio of digits and punctuation to words is greater than 0.25."""
    words = re.findall(r'\w+', text)
    if not words:
        return False
    digits_and_punct = sum(1 for ch in text if ch.isdigit() or ch in string.punctuation)
    return (digits_and_punct / len(words)) > 0.25

def no_special_characters(text):
    """Assume that if curly braces are present, it might be code.
       Return True if no '{' or '}' is found.
    """
    return "{" not in text and "}" not in text

def terminal_punctuation(text):
    """Check if the line ends with ., !, ?, or a quote."""
    text = text.strip()
    return text.endswith('.') or text.endswith('!') or text.endswith('?') or text.endswith('"')

def stop_word_match_2(text):
    """Check if the text contains at least 2 stopwords from our set."""
    words = re.findall(r'\w+', text.lower())
    count = sum(1 for word in words if word in STOPWORDS)
    return count >= 2

def javascript_flag(text):
    """Return True if the text contains phrases indicative of code."""
    return ("javascript" in text.lower()) or ("lorem ipsum" in text.lower())

def token_count_ge_3(text):
    """Check if token count is at least 3."""
    doc = nlp(text)
    return len(doc) >= 3

def word_count_3_256(text):
    """Check if word count is between 3 and 256."""
    words = re.findall(r'\w+', text)
    return 3 <= len(words) <= 256

def has_object(text):
    """Check if the sentence contains an object using dependency parsing."""
    doc = nlp(text)
    for token in doc:
        # looking for direct object or object of a preposition
        if token.dep_ in {"dobj", "obj", "pobj"}:
            return True
    return False

def has_noun(text):
    """Check if there is at least one noun in the sentence."""
    doc = nlp(text)
    return any(token.pos_ == "NOUN" for token in doc)

def has_determiner(text):
    """Check if the sentence contains a determiner."""
    doc = nlp(text)
    return any(token.pos_ == "DET" for token in doc)

def text_complexity_c1(text):
    """A simple heuristic for text complexity.
       For example, we flag text as complex if it has an object.
       (This can be made more elaborate by checking parse tree depth.)
    """
    return has_object(text)

# Map heuristic names to functions
heuristics = {
    "has_first_letter_caps": has_first_letter_caps,
    "no_all_caps": no_all_caps,
    "word_repetition_ratio_ge_0_2": word_repetition_ratio_ge_0_2,
    "digit_punctuation_ratio": digit_punctuation_ratio,
    "no_special_characters": no_special_characters,
    "terminal_punctuation": terminal_punctuation,
    "stop_word_match_2": stop_word_match_2,
    "javascript_flag": javascript_flag,
    "token_count_ge_3": token_count_ge_3,
    "word_count_3_256": word_count_3_256,
    "has_object": has_object,
    "has_noun": has_noun,
    "has_determiner": has_determiner,
    "text_complexity_c1": text_complexity_c1,
}

# ---------------------------
# Perplexity Calculation (Optional)
# ---------------------------
def calculate_perplexity(text, model, tokenizer, device="cpu"):
    """
    Calculate the perplexity of a given text using a pretrained language model.
    Note: This function uses a causal LM model from HuggingFace.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())

# ---------------------------
# Weight Computation for Heuristics
# ---------------------------
def compute_heuristic_weights(dataset_lines, model, tokenizer, device="cpu"):
    """
    Compute weights for each heuristic filter using the formulation:
      w_i = max(0, (PPL_all - PPL_i) / PPL_all)
    where PPL_all is the perplexity for the unfiltered dataset and
    PPL_i is the perplexity for the subset of lines that satisfy heuristic i.
    """
    # Join all lines to compute overall perplexity
    all_text = "\n".join(dataset_lines)
    PPL_all = calculate_perplexity(all_text, model, tokenizer, device)
    weights = {}
    for name, func in heuristics.items():
        # Filter lines that satisfy the heuristic
        filtered_lines = [line for line in dataset_lines if func(line)]
        if not filtered_lines:
            weights[name] = 0.0
            continue
        filtered_text = "\n".join(filtered_lines)
        PPL_i = calculate_perplexity(filtered_text, model, tokenizer, device)
        # Compute weight (ensure non-negative)
        weights[name] = max(0.0, (PPL_all - PPL_i) / PPL_all)
    return weights

# ---------------------------
# Quality Scoring Functions
# ---------------------------
def compute_line_quality(text, weights):
    """
    Compute quality score for a single line using:
      score_line = sum_i (w_i * I_i(text)) / sum_i (w_i)
    where I_i(text) is 1 if the line passes heuristic i, else 0.
    """
    numerator = 0.0
    denominator = 0.0
    for name, func in heuristics.items():
        w = weights.get(name, 0.0)
        indicator = 1 if func(text) else 0
        numerator += w * indicator
        denominator += w
    if denominator == 0:
        return 0.0
    return numerator / denominator

def compute_document_quality(document, weights):
    """
    Compute a document-level quality score by splitting the document into lines,
    computing each line’s quality score, and then taking a weighted average of
    these scores using token counts as weights.
    """
    # Here we simply split by newline; more sophisticated sentence splitting can be used.
    lines = [line for line in document.splitlines() if line.strip()]
    total_tokens = 0
    weighted_score = 0.0
    for line in lines:
        doc_line = nlp(line)
        token_count = len(doc_line)
        line_score = compute_line_quality(line, weights)
        weighted_score += token_count * line_score
        total_tokens += token_count
    if total_tokens == 0:
        return 0.0
    return weighted_score / total_tokens

# ---------------------------
# Data Pruning Based on Quality Scores
# ---------------------------
def prune_dataset(dataset, weights, percentile_threshold=80):
    """
    Prune the dataset by computing the document quality score for each document
    and retaining only those with scores in the top percentile_threshold.
    Here, dataset is a list of documents (strings).
    """
    # Compute quality scores for each document
    scored_docs = [(doc, compute_document_quality(doc, weights)) for doc in dataset]
    # Determine the quality score threshold (e.g., top 20% if threshold=80)
    scores = [score for _, score in scored_docs]
    threshold_value = np.percentile(scores, percentile_threshold)
    # Retain documents with quality scores above the threshold
    pruned_docs = [doc for doc, score in scored_docs if score >= threshold_value]
    return pruned_docs

    