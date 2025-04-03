import os
import json
import spacy
import pytextrank
import numpy as np
from pathlib import Path
import subprocess

# Ensure the spaCy model is installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Model 'en_core_web_sm' not found. Installing...")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("textrank")

def evaluate_coherence(text):
    """
    Evaluate the coherence of a text using TextRank.
    Returns the average rank of phrases as a coherence measure.
    """
    try:
        doc = nlp(text)
        
        # Extract the phrases and their ranks
        phrases = list(doc._.phrases)
        
        # If there are no phrases, return a low coherence score
        if not phrases:
            return 0.0
        
        # Calculate average phrase rank as a coherence measure
        avg_rank = np.mean([phrase.rank for phrase in phrases])
        return float(avg_rank)
    except Exception as e:
        print(f"Error evaluating coherence: {e}")
        return 0.0

def process_iterations_file(file_path):
    """
    Process a JSON file containing iterations and evaluate coherence.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = []
    
    for item in data:
        item_id = item.get('id')
        original_text = item.get('original_text', '')
        summarized_text = item.get('summarized_text', '')
        iterations = item.get('iterations', [])
        
        # Evaluate coherence of summarized text
        summarized_coherence = evaluate_coherence(summarized_text)
        
        # Evaluate coherence of each iteration
        iteration_coherence_scores = []
        for iteration in iterations:
            if isinstance(iteration, dict) and 'rewritten_text' in iteration:
                text = iteration.get('rewritten_text', '')
                if text.startswith("Error processing:") or text.startswith("Encountered an error:"):
                    # Skip error messages
                    continue
                coherence = evaluate_coherence(text)
                iteration_coherence_scores.append(coherence)
        
        # Create result object matching BLEU files format exactly
        result = {
            'id': item_id,
            'paraphrased_score': summarized_coherence,
            'iterations_scores': iteration_coherence_scores,
            'sentiment_accuracy': 1.0  # Placeholder to match BLEU format
        }
        
        results.append(result)
    
    return results

def main():
    # Define base path
    base_path = Path('f:/Programming/GitHub/chameleon-org/chameleon-testing/tools/iterativeSampling/')
    
    # Define known iteration file locations
    iteration_files = {
        'gemini2Flash': base_path / '2_iterativePrompting/gemini2FlashIteration/gemini2FlashIterations.json',
    }
    
    # Search for other iteration files
    iteration_dirs = list((base_path / '2_iterativePrompting').glob('*Iteration'))
    for iteration_dir in iteration_dirs:
        model_name = iteration_dir.name.replace('Iteration', '')
        json_files = list(iteration_dir.glob('*Iterations.json'))
        if json_files:
            iteration_files[model_name] = json_files[0]
    
    # Create output directory
    output_dir = base_path / '6_Coherence'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model's iterations
    for model_name, file_path in iteration_files.items():
        if file_path.exists():
            print(f"Processing {model_name}...")
            results = process_iterations_file(file_path)
            
            # Save results (only numerical scores, similar to BLEU files)
            output_file = output_dir / f"{model_name}_coherence.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        else:
            print(f"Warning: File not found: {file_path}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
