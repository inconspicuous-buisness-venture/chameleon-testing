from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import json

def read_jsons_from_subfolders(base_folder):
    """Read all JSON files from subfolders."""
    json_data = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        json_data.append((file_path, data))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON from {file_path}: {e}")
    return json_data

def is_error_message(text):
    """Check if the text is an error message from the API."""
    error_patterns = [
        "Error processing: 429",
        "Error: 429",
        "RESOURCE_EXHAUSTED",
        "You exceeded your current quota",
        "Error occurred"
    ]
    
    if isinstance(text, str):
        return any(pattern in text for pattern in error_patterns)
    return False

def calculate_real_perplexity(text, tokenizer, model, device):
    """Calculate actual perplexity using GPT-2 model."""
    if not isinstance(text, str) or text.strip() == "":
        return None

    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    max_length = model.config.n_positions

    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
    
    return perplexity

def calculate_real_perplexity_for_all():
    """Calculate real perplexity scores for all models' iterations."""
    base_folder = "/Users/makaip/Documents/GitHub/chameleon-testing/tools/iterativeSampling/2_iterativePrompt"

    tokenizer, model, device = setup_real_perplexity()
    all_json_data = read_jsons_from_subfolders(base_folder)

    for file_path, data in all_json_data:
        if "gemini2Flash" in file_path:
            model_name = "gemini2Flash"
        elif "gemini15Pro" in file_path:
            model_name = "gemini15Pro"
        elif "GPT4o" in file_path:
            model_name = "GPT4o"
        else:
            continue
            
        if not (isinstance(data, list) and len(data) > 0 and "id" in data[0]):
            print(f"Skipping {file_path}: Invalid data format")
            continue
            
        perplexity_results = []

        for item in data:
            item_id = item.get("id")
            if not item_id:
                continue

            summarized_text = item.get("summarized_text", "")
            summarized_perplexity = calculate_real_perplexity(summarized_text, tokenizer, model, device)

            iterations_perplexity = []
            iterations = item.get("iterations", [])
            for iteration in iterations:
                rewritten_text = iteration.get("rewritten_text", "")
                if not is_error_message(rewritten_text):
                    perplexity = calculate_real_perplexity(rewritten_text, tokenizer, model, device)
                    iterations_perplexity.append(perplexity)
                else:
                    iterations_perplexity.append(None)

            result = {
                "id": item_id,
                "paraphrased_perplexity": summarized_perplexity,
                "iterations_perplexity": iterations_perplexity
            }
            perplexity_results.append(result)

        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            f"{model_name}Iterations_perplexity.json"
        )
        with open(output_path, 'w') as f:
            json.dump(perplexity_results, f, indent=2)
        
        print(f"Real perplexity scores for {model_name} saved to {output_path}")
        print(f"Total items processed for {model_name}: {len(perplexity_results)}")

def setup_real_perplexity():
    """Set up the model for calculating real perplexity."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device

if __name__ == "__main__":
    calculate_real_perplexity_for_all()
