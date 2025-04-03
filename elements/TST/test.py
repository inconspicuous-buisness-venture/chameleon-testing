import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# BLEU Score Calculation
def calculate_bleu_scores(original, paraphrased, iterations):
    smoothing = SmoothingFunction().method1
    return {
        "paraphrased_score": sentence_bleu([original.split()], paraphrased.split(), smoothing_function=smoothing),
        "iterations_scores": [
            sentence_bleu([original.split()], iteration["rewritten_text"].split(), smoothing_function=smoothing)
            for iteration in iterations
        ]
    }

# Sentiment Accuracy Calculation with automatic truncation
def sentiment_accuracy(predictions, references):
    correct = sum(
        sentiment_analyzer(pred, truncation=True)[0]['label'] == sentiment_analyzer(ref, truncation=True)[0]['label']
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)

# File paths
input_filepath = r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\2_iterativePrompting\GPT4oIteration\GPT4oIterations.json"
output_filepath = r"f:\Programming\GitHub\chameleon-org\chameleon-testing\tools\iterativeSampling\5_BLEUandSentiment\GPT4oIterations_bleu.json"

# Read input JSON file
with open(input_filepath, "r", encoding="utf-8") as file:
    data = json.load(file)

# Process data and calculate BLEU scores & sentiment accuracy
results = []
for entry in data:
    original_text = entry["original_text"]
    paraphrased_text = entry["summarized_text"]
    iterations = entry["iterations"]
    
    bleu_scores = calculate_bleu_scores(original_text, paraphrased_text, iterations)
    sent_acc = sentiment_accuracy([paraphrased_text], [original_text])
    
    results.append({
        "id": entry["id"],
        "paraphrased_score": bleu_scores["paraphrased_score"],
        "iterations_scores": bleu_scores["iterations_scores"],
        "sentiment_accuracy": sent_acc
    })

# Write results to output JSON file
with open(output_filepath, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

# Print completion message
print(f"Processing complete. Results saved to {output_filepath}.")