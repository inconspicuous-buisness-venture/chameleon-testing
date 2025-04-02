import json
import time
import os
import logging
from datetime import datetime
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"roberta_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Set up logging to file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add console handler to see logs on screen as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Load RoBERTa model and tokenizer
logging.info("Loading RoBERTa model and tokenizer...")
print("Loading RoBERTa model and tokenizer... This may take a moment.")

# Global variables for model and tokenizer to avoid reloading
tokenizer = None
model = None

def load_roberta_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
            model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
            logging.info("RoBERTa model and tokenizer loaded successfully")
            print("RoBERTa model loaded successfully!")
        except Exception as e:
            logging.error(f"Error loading RoBERTa model: {e}")
            print(f"Error loading RoBERTa model: {e}")
            raise

def check_text_with_roberta(text):
    """Check if text is AI-generated using RoBERTa model"""
    global tokenizer, model
    
    # Ensure model is loaded
    if tokenizer is None or model is None:
        load_roberta_model()
    
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the logits (output scores before activation)
        logits = outputs.logits
        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)
        
        # Index 0: Not GPT-generated, Index 1: GPT-generated
        # Return as percentage for consistency with previous code
        prob_gpt_generated = (1 - probabilities[0][1].item()) * 100
        
        return prob_gpt_generated
    except Exception as e:
        logging.error(f"RoBERTa detection error: {e}")
        print(f"Error during text detection: {e}")
        return None

def save_temp_results(data, temp_file, final=False):
    """Save current results to a temporary file to prevent data loss"""
    try:
        temp_dir = os.path.dirname(temp_file)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            
        # First write to a temporary file, then rename to avoid corruption if interrupted
        temp_temp_file = f"{temp_file}.tmp"
        with open(temp_temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Safely replace the previous temp file
        if os.path.exists(temp_file):
            os.replace(temp_temp_file, temp_file)
        else:
            os.rename(temp_temp_file, temp_file)
            
        if final:
            logging.info(f"Final results saved to: {temp_file}")
        else:
            logging.debug(f"Temporary results updated in: {temp_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving temporary results: {e}")
        return False

def process_iterations_data(input_file, output_file):
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            print(f"Error: Input file not found: {input_file}")
            return
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):  # Fix: changed os.exists to os.path.exists
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        
        # Set up temporary file
        temp_dir = os.path.join(os.path.dirname(output_file), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"temp_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logging.info(f"Temporary results will be saved to: {temp_file}")
        
        # Load the input JSON data
        logging.info(f"Loading data from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_entries = len(data)
        logging.info(f"Processing {total_entries} entries...")
        print(f"Processing {total_entries} entries with RoBERTa detection...")
        
        # Make sure model is loaded before starting
        load_roberta_model()
        
        # Save initial state to temp file
        save_temp_results(data, temp_file)
        
        # Process each entry
        for idx, entry in enumerate(data):
            entry_id = entry['id']
            progress = (idx / total_entries) * 100
            logging.info(f"Processing entry {idx+1}/{total_entries} (ID: {entry_id}) - {progress:.1f}% complete")
            print(f"[{progress:.1f}%] Processing entry ID {entry_id}...")
            
            # Check original text
            logging.info(f"  Checking original text for entry {entry_id}...")
            original_text = entry['original_text']
            logging.debug(f"Original text: {original_text[:100]}...")
            
            original_score = check_text_with_roberta(original_text)
            logging.info(f"  Original text score: {original_score}")
            
            # Add metrics to the entry - check if it already exists
            if 'original_text_metrics' not in entry:
                entry['original_text_metrics'] = {
                    "detection_scores": {
                        "roberta": original_score
                    }
                }
            else:
                # Add or update the roberta score in the existing detection_scores
                if "detection_scores" not in entry['original_text_metrics']:
                    entry['original_text_metrics']["detection_scores"] = {}
                entry['original_text_metrics']["detection_scores"]["roberta"] = original_score
            
            # Save after each original text processing
            save_temp_results(data, temp_file)
            
            # Check summarized text
            logging.info(f"  Checking summarized text for entry {entry_id}...")
            summarized_text = entry['summarized_text']
            logging.debug(f"Summarized text: {summarized_text[:100]}...")
            
            summarized_score = check_text_with_roberta(summarized_text)
            logging.info(f"  Summarized text score: {summarized_score}")
            
            # Add metrics to the entry - check if it already exists
            if 'summarized_text_metrics' not in entry:
                entry['summarized_text_metrics'] = {
                    "detection_scores": {
                        "roberta": summarized_score
                    }
                }
            else:
                # Add or update the roberta score in the existing detection_scores
                if "detection_scores" not in entry['summarized_text_metrics']:
                    entry['summarized_text_metrics']["detection_scores"] = {}
                entry['summarized_text_metrics']["detection_scores"]["roberta"] = summarized_score
            
            # Save after summarized text processing
            save_temp_results(data, temp_file)
            
            # Process each iteration
            iterations_count = len(entry['iterations'])
            logging.info(f"  Processing {iterations_count} iterations for entry {entry_id}...")
            
            for i, iteration in enumerate(entry['iterations']):
                iter_progress = ((i+1) / iterations_count) * 100
                logging.info(f"  Checking iteration {i+1}/{iterations_count} ({iter_progress:.1f}%)...")
                print(f"    [{progress:.1f}% main] Iteration {i+1}/{iterations_count} ({iter_progress:.1f}%)...")
                
                # Some iterations might have an error message instead of rewritten_text
                if 'rewritten_text' in iteration:
                    # Log the text being checked
                    rewritten_text = iteration['rewritten_text']
                    logging.debug(f"  Iteration {i+1} text: {rewritten_text[:100]}...")
                    
                    # Check the rewritten text
                    score = check_text_with_roberta(rewritten_text)
                    logging.info(f"  Iteration {i+1} score: {score}")
                    
                    # Add the iteration number if it doesn't exist
                    if 'iteration_number' not in iteration:
                        iteration['iteration_number'] = i + 1
                    
                    # Add or update the detection score
                    if 'detection_scores' not in iteration:
                        iteration['detection_scores'] = {}
                    iteration['detection_scores']["roberta"] = score
                else:
                    # Handle missing rewritten_text (error cases)
                    logging.warning(f"  Iteration {i+1} has no rewritten_text")
                    
                    # Add the iteration number if it doesn't exist
                    if 'iteration_number' not in iteration:
                        iteration['iteration_number'] = i + 1
                    
                    # Add or update the detection score
                    if 'detection_scores' not in iteration:
                        iteration['detection_scores'] = {}
                    iteration['detection_scores']["roberta"] = None
                
                # Save after each iteration
                if (i + 1) % 3 == 0 or i == iterations_count - 1:  # Save every 3 iterations or at the end
                    save_temp_results(data, temp_file)
                
                # Add a small delay to avoid system resource issues
                time.sleep(0.1)
            
            # Save after completing all iterations for this entry
            logging.info(f"Completed processing for entry ID {entry_id}")
            save_temp_results(data, temp_file)
        
        # Save the processed data to the output file
        logging.info(f"All entries processed. Saving final results to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Make a backup copy of the temp file with a "final" suffix
        final_temp = os.path.join(temp_dir, f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(temp_file, final_temp)
        
        logging.info(f"Processing complete. Results saved to {output_file} and {final_temp}")
        print(f"Processing complete! Results saved to {output_file}")
        print(f"Log file created at: {log_file}")
        print(f"Backup of final results saved to: {final_temp}")
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Partial results are saved in the temporary file.")
        print(f"\nProcess interrupted. Partial results available at: {temp_file}")
        return
    except Exception as e:
        logging.error(f"Error processing data: {e}", exc_info=True)
        print(f"Error processing data: {e}")
        print(f"Partial results may be available in: {temp_file}")

def get_file_path(prompt, default_path=None, must_exist=False):
    """Ask user for a file path with a default option"""
    if default_path:
        user_input = input(f"{prompt} [default: {default_path}]: ")
        if not user_input.strip():
            return default_path
        path = user_input
    else:
        path = input(f"{prompt}: ")
    
    # Check if file exists if required
    if must_exist and not os.path.exists(path):
        print(f"Error: The specified file does not exist: {path}")
        return get_file_path(prompt, default_path, must_exist)
    
    return path

if __name__ == "__main__":
    print("RoBERTa AI Detection Tool")
    print("========================")
    
    # Default paths
    default_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "2_iterativePrompting", "gemini2FlashIteration", "gemini2FlashIterations.json")
    
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "gemini2FlashIterationsWithRoBERTaScores.json")
    
    # Get input and output file paths from user
    input_path = get_file_path("Enter the path to the input JSON file", default_input, must_exist=True)
    output_path = get_file_path("Enter the path for the output JSON file", default_output)
    
    # Process the data
    logging.info("Starting RoBERTa detection process")
    process_iterations_data(input_path, output_path)
    logging.info("RoBERTa detection process completed")
