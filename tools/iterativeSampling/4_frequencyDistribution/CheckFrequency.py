import json
import os
import logging
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
import io
import base64

# Configure logging
log_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, f"frequency_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

def get_word_frequencies(text):
    """
    Compute word frequencies from text
    Returns both the raw counts and normalized distribution
    """
    if not text or not isinstance(text, str):
        return {}, {}
    
    # Normalize and split text
    words = text.lower().split()
    
    # Skip if no words
    if not words:
        return {}, {}
    
    # Get raw counts
    word_counts = Counter(words)
    
    # Get frequency distribution (how many times each frequency appears)
    frequency_values = list(word_counts.values())
    frequency_distribution = Counter(frequency_values)
    
    # Convert to dictionaries with string keys for JSON serialization
    frequency_distribution_dict = {str(k): v for k, v in frequency_distribution.items()}
    word_counts_dict = {str(k): v for k, v in word_counts.items()}
    
    return word_counts_dict, frequency_distribution_dict

def generate_density_plot(text, title="Word Frequency Distribution"):
    """Generate a KDE plot for word frequencies and return as base64 encoded string"""
    if not text or not isinstance(text, str):
        return None
    
    words = text.lower().split()
    if not words:
        return None
        
    word_counts = Counter(words)
    freq_values = list(word_counts.values())
    
    if len(freq_values) < 2:  # Need at least 2 points for KDE
        return None
    
    # Create plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(freq_values, fill=True)
    plt.title(title)
    plt.xlabel('Word Frequency')
    plt.ylabel('Density')
    
    # Save plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode the image
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

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

def process_frequency_data(input_file, output_file=None):
    """
    Process data and append frequency distribution metrics.
    If output_file is None, will overwrite the input_file.
    """
    # Initialize temp_file variable outside the try block to avoid UnboundLocalError
    temp_file = None
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            print(f"Error: Input file not found: {input_file}")
            return
        
        # If no output file is specified, modify the input file directly
        if output_file is None:
            output_file = input_file
            logging.info(f"Will modify input file directly: {input_file}")
            print(f"Will modify input file directly: {input_file}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):  # Fixed: os.exists -> os.path.exists
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")
        
        # Set up temporary file
        temp_dir = os.path.join(os.path.dirname(output_file), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"temp_freq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        logging.info(f"Temporary results will be saved to: {temp_file}")
        
        # Load the input JSON data
        logging.info(f"Loading data from: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_entries = len(data)
        logging.info(f"Processing {total_entries} entries...")
        print(f"Processing {total_entries} entries for word frequency analysis...")
        
        # Save initial state to temp file
        save_temp_results(data, temp_file)
        
        # Process each entry
        for idx, entry in enumerate(data):
            entry_id = entry.get('id', idx)
            progress = (idx / total_entries) * 100
            logging.info(f"Processing entry {idx+1}/{total_entries} (ID: {entry_id}) - {progress:.1f}% complete")
            print(f"[{progress:.1f}%] Processing entry ID {entry_id}...")
            
            # Process original text - this matches your specific JSON schema
            if 'originalText' in entry:
                original_text_obj = entry['originalText']
                if 'text' in original_text_obj:
                    original_text = original_text_obj['text']
                    logging.info(f"  Analyzing word frequencies in original text for entry {entry_id}...")
                    
                    # Get word frequency data
                    word_counts, frequency_distribution = get_word_frequencies(original_text)
                    density_plot = generate_density_plot(original_text, f"Word Frequency - Original Text (ID: {entry_id})")
                    
                    # Add or update word_frequency in the originalText object
                    original_text_obj["word_frequency"] = {
                        "density": word_counts,
                        "frequency_distribution": frequency_distribution
                    }
                    
                    if density_plot:
                        original_text_obj["word_frequency"]["density_plot"] = density_plot
                    
                    # Save after processing
                    save_temp_results(data, temp_file)
            
            # Process models section that contains summarizedText and iterations
            if 'models' in entry:
                for model_name, model_data in entry['models'].items():
                    logging.info(f"  Processing model: {model_name} for entry {entry_id}...")
                    
                    # Process summarizedText if it exists
                    if 'summarizedText' in model_data and 'text' in model_data['summarizedText']:
                        summarized_text_obj = model_data['summarizedText']
                        summarized_text = summarized_text_obj['text']
                        logging.info(f"  Analyzing word frequencies in summarized text for model {model_name}, entry {entry_id}...")
                        
                        # Get word frequency data
                        word_counts, frequency_distribution = get_word_frequencies(summarized_text)
                        density_plot = generate_density_plot(
                            summarized_text, 
                            f"Word Frequency - Summarized Text - {model_name} (ID: {entry_id})"
                        )
                        
                        # Add or update word_frequency in the summarizedText object
                        summarized_text_obj["word_frequency"] = {
                            "density": word_counts,
                            "frequency_distribution": frequency_distribution
                        }
                        
                        if density_plot:
                            summarized_text_obj["word_frequency"]["density_plot"] = density_plot
                        
                        # Save after processing each model's summarized text
                        save_temp_results(data, temp_file)
                    
                    # Process iterations
                    if 'iterations' in model_data:
                        iterations = model_data['iterations']
                        iterations_count = len(iterations)
                        logging.info(f"  Processing {iterations_count} iterations for model {model_name}, entry {entry_id}...")
                        
                        for i, iteration in enumerate(iterations):
                            iter_progress = ((i+1) / iterations_count) * 100
                            logging.info(f"  Checking iteration {i+1}/{iterations_count} ({iter_progress:.1f}%)...")
                            print(f"    [{progress:.1f}% main] Model {model_name} - Iteration {i+1}/{iterations_count} ({iter_progress:.1f}%)...")
                            
                            # Ensure the iteration has text
                            if 'text' in iteration:
                                iter_text = iteration['text']
                                
                                # Get word frequency data
                                word_counts, frequency_distribution = get_word_frequencies(iter_text)
                                density_plot = generate_density_plot(
                                    iter_text, 
                                    f"Word Frequency - {model_name} - Iteration {i+1} (ID: {entry_id})"
                                )
                                
                                # Add the iteration number if it doesn't exist
                                if 'iteration' not in iteration:
                                    iteration['iteration'] = i + 1
                                
                                # Add or update word frequency metrics
                                iteration["word_frequency"] = {
                                    "density": word_counts,
                                    "frequency_distribution": frequency_distribution
                                }
                                
                                if density_plot:
                                    iteration["word_frequency"]["density_plot"] = density_plot
                            else:
                                # Handle missing text (error cases)
                                logging.warning(f"  Iteration {i+1} has no text field")
                                
                                # Add the iteration number if it doesn't exist
                                if 'iteration' not in iteration:
                                    iteration['iteration'] = i + 1
                                
                                # Add empty word frequency metrics
                                iteration["word_frequency"] = {
                                    "density": {},
                                    "frequency_distribution": {}
                                }
                            
                            # Save after batches of iterations
                            if (i + 1) % 5 == 0 or i == iterations_count - 1:
                                save_temp_results(data, temp_file)
            
            # Save after completing all models and iterations for this entry
            logging.info(f"Completed processing for entry ID {entry_id}")
            save_temp_results(data, temp_file)
        
        # Save the processed data to the output file
        logging.info(f"All entries processed. Saving final results to: {output_file}")
        
        # Create a backup of the original input file if we're overwriting it
        if input_file == output_file:
            backup_file = f"{input_file}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logging.info(f"Creating backup of original file: {backup_file}")
            shutil.copy2(input_file, backup_file)
            print(f"Backup of original file created at: {backup_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Make a backup copy of the temp file with a "final" suffix
        final_temp = os.path.join(temp_dir, f"final_freq_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(temp_file, final_temp)
        
        logging.info(f"Processing complete. Results saved to {output_file}")
        print(f"Processing complete! Results saved to {output_file}")
        print(f"Log file created at: {log_file}")
        print(f"Backup of final results saved to: {final_temp}")
    
    except KeyboardInterrupt:
        logging.warning("Process interrupted by user. Partial results are saved in the temporary file.")
        if temp_file:  # Only print if temp_file has been defined
            print(f"\nProcess interrupted. Partial results available at: {temp_file}")
        else:
            print("\nProcess interrupted before temporary file was created.")
        return
    except Exception as e:
        logging.error(f"Error processing data: {e}", exc_info=True)
        print(f"Error processing data: {e}")
        if temp_file:  # Only print if temp_file has been defined
            print(f"Partial results may be available in: {temp_file}")
        return

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

def compare_frequency_distributions(text1, text2, title1="Text 1", title2="Text 2", output_file=None):
    """
    Compare word frequency distributions between two texts and save or show the plot
    """
    if not text1 or not text2 or not isinstance(text1, str) or not isinstance(text2, str):
        return None
    
    # Get word frequencies
    words1 = text1.lower().split()
    words2 = text2.lower().split()
    
    if not words1 or not words2:
        return None
    
    word_counts1 = Counter(words1)
    word_counts2 = Counter(words2)
    
    freq_values1 = list(word_counts1.values())
    freq_values2 = list(word_counts2.values())
    
    if len(freq_values1) < 2 or len(freq_values2) < 2:  # Need at least 2 points for KDE
        return None
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot both distributions
    sns.kdeplot(freq_values1, fill=True, color='blue', alpha=0.5, label=title1)
    sns.kdeplot(freq_values2, fill=True, color='red', alpha=0.5, label=title2)
    
    plt.title(f"Word Frequency Distribution Comparison: {title1} vs {title2}")
    plt.xlabel('Word Frequency')
    plt.ylabel('Density')
    plt.legend()
    
    if output_file:
        # Save to file
        plt.savefig(output_file)
        plt.close()
        return output_file
    else:
        # Save to buffer for display
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_str

if __name__ == "__main__":
    print("Word Frequency Distribution Analysis Tool")
    print("========================================")
    
    # Default paths
    default_input = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "3_checkingHumanity", "roberta_results", "gemini2FlashIterations_with_roberta.json")
    
    # Get input file path
    input_path = get_file_path("Enter the path to the JSON file", default_input, must_exist=True)
    
    # Ask if user wants to modify directly or create a new file
    modify_directly = input("Do you want to modify the input file directly? (y/n) [default: y]: ").strip().lower()
    
    if modify_directly == "" or modify_directly.startswith("y"):
        output_path = None  # Signal to modify input file directly
        print(f"Will modify the input file directly: {input_path}")
    else:
        # Get output file path
        default_output = os.path.join(os.path.dirname(input_path), 
                             f"{os.path.splitext(os.path.basename(input_path))[0]}_with_frequency.json")
        output_path = get_file_path("Enter the path for the output JSON file", default_output)
    
    # Process the data
    logging.info("Starting word frequency analysis process")
    process_frequency_data(input_path, output_path)
    logging.info("Word frequency analysis process completed")
