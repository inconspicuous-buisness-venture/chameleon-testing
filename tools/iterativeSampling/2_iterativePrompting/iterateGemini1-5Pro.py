import csv
import os
import time
import pandas as pd
from google import genai
import sys
import logging
import datetime

def setup_logging(log_file):
    """Configure logging to both console and file."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def read_prompt_template(file_path):
    """Read the prompt template from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            template = f.read()
        return template
    except Exception as e:
        print(f"Error reading prompt template: {e}")
        return None

def read_csv_column(file_path, column_name=None, column_index=None):
    """Read the specified column from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        
        # Decide whether to use column_name or column_index
        if column_name is not None:
            if column_name not in df.columns:
                print(f"Column '{column_name}' not found. Available columns: {df.columns.tolist()}")
                return []
            return df[column_name].tolist()
        elif column_index is not None:
            if column_index >= len(df.columns):
                print(f"Column index {column_index} is out of range. Using first column.")
                column_index = 0
            return df.iloc[:, column_index].tolist()
        else:
            # Default to first column
            return df.iloc[:, 0].tolist()
            
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def process_with_gemini(prompt, client):
    """Send the prompt to Gemini API and get the response."""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-pro", 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return f"Error processing: {str(e)}"

def save_results(results, output_file, append=False):
    """Save results to a CSV file."""
    mode = 'a' if append else 'w'
    try:
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(["Original Text", "Humanized Text"])
            writer.writerows(results)
        print(f"Results successfully saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def process_batch(texts, start_idx, end_idx, prompt_template, client, output_csv, batch_num, all_results, is_first_batch_of_iteration, logger):
    """Process a batch of texts."""
    batch_start_time = time.time()
    logger.info(f"Starting batch processing for texts {start_idx+1}-{end_idx} of {len(texts)}")
    
    batch_results = []
    # Process each text in current batch
    for i, text in enumerate(texts[start_idx:end_idx]):
        current_idx = start_idx + i
        logger.info(f"Processing text {current_idx+1}/{len(texts)}...")
        
        try:
            # Prepare the prompt by replacing placeholder with the text
            prompt_start_time = time.time()
            full_prompt = prompt_template.replace("Here is the text:", f"Here is the text:\n{text}")
            prompt_prep_time = time.time() - prompt_start_time
            logger.debug(f"Prompt preparation took {prompt_prep_time:.2f} seconds")
            
            # Send to Gemini API
            api_start_time = time.time()
            humanized_text = process_with_gemini(full_prompt, client)
            api_time = time.time() - api_start_time
            logger.debug(f"API call took {api_time:.2f} seconds")
            
            # Store result
            batch_results.append((text, humanized_text))
            all_results.append((text, humanized_text))
            
            # Add a small delay between individual requests
            if i < end_idx - start_idx - 1:
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error processing text {current_idx+1}: {e}")
            # Save what we have so far if there's an error
            save_results(all_results, output_csv, not is_first_batch_of_iteration or batch_num > 0)
            return False, all_results
    
    # Batch complete - save results
    save_results(batch_results, output_csv, not is_first_batch_of_iteration or batch_num > 0)
    batch_time = time.time() - batch_start_time
    logger.info(f"Batch processing complete. Took {batch_time:.2f} seconds")
    return True, all_results

def get_iteration_output_path(base_path, iteration):
    """Generate the output path for a specific iteration."""
    dir_path = os.path.dirname(base_path)
    filename = f"iteration{iteration}.csv"
    return os.path.join(dir_path, filename)

def run_iteration(input_csv, output_csv, prompt_template, client, iteration_num, logger):
    """Run a single iteration of the text processing."""
    logger.info(f"\n{'='*40}")
    logger.info(f"STARTING ITERATION {iteration_num}")
    logger.info(f"{'='*40}")
    
    # Determine which column to read based on iteration
    if iteration_num == 1:
        # First iteration uses "Rewritten Text" column from the original input
        texts = read_csv_column(input_csv, column_name="Rewritten Text")
    else:
        # Subsequent iterations use the first column (Humanized Text from previous iteration)
        texts = read_csv_column(input_csv, column_index=0)
    
    if not texts:
        logger.error("No texts found in the CSV.")
        return False
    
    logger.info(f"Found {len(texts)} texts to process.")
    
    # Process texts in batches
    batch_size = 15
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    all_results = []
    
    try:
        for batch_num in range(total_batches):
            logger.info(f"\nProcessing batch {batch_num+1}/{total_batches}...")
            
            # Calculate batch start and end
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            
            # Process the batch
            success, all_results = process_batch(
                texts, start_idx, end_idx, prompt_template, client, output_csv, 
                batch_num, all_results, is_first_batch_of_iteration=(batch_num==0), logger=logger
            )
            
            if not success:
                return False
            
            # Wait a minute before the next batch (unless it's the last batch)
            if batch_num < total_batches - 1:
                logger.info(f"Batch {batch_num+1} complete. Waiting 60 seconds before next batch...")
                time.sleep(60)
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        return False
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        return False
    
    logger.info(f"Iteration {iteration_num} completed successfully!")
    return True

def prepare_for_next_iteration(current_output_csv):
    """Prepare the output CSV for the next iteration by extracting only the humanized text column."""
    try:
        # Read the current output
        df = pd.read_csv(current_output_csv)
        
        # Get only the "Humanized Text" column
        humanized_texts = df["Humanized Text"]
        
        # Create a new DataFrame with just this column 
        new_df = pd.DataFrame({"Text": humanized_texts})
        
        # Save to the same file (overwriting it)
        new_df.to_csv(current_output_csv, index=False)
        
        print(f"Prepared {current_output_csv} for next iteration.")
        return True
    except Exception as e:
        print(f"Error preparing for next iteration: {e}")
        return False

def main():
    # Initialize Gemini API client
    api_key = "AIzaSyC3l2HJFZIYvbrZE7jVZAn1qswWHhkC9kU"
    client = genai.Client(api_key=api_key)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(script_dir, "humanizationPrompt.txt")
    
    # Get initial input
    initial_input_csv = input("Enter path to initial input CSV file: ")
    output_dir = input("Enter directory for output files: ")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, f"log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    logger = setup_logging(log_file)
    
    # Read the prompt template
    prompt_template = read_prompt_template(prompt_template_path)
    if not prompt_template:
        logger.error("Failed to load prompt template.")
        return
    
    # Run 10 iterations
    total_iterations = 10
    current_input_csv = initial_input_csv
    
    for iteration in range(1, total_iterations + 1):
        # Generate output path for this iteration
        iteration_output_csv = os.path.join(output_dir, f"iteration{iteration}.csv")
        
        logger.info(f"\nStarting iteration {iteration}/{total_iterations}")
        logger.info(f"Input: {current_input_csv}")
        logger.info(f"Output: {iteration_output_csv}")
        
        # Run this iteration
        success = run_iteration(
            input_csv=current_input_csv,
            output_csv=iteration_output_csv,
            prompt_template=prompt_template,
            client=client,
            iteration_num=iteration,
            logger=logger
        )
        
        if not success:
            logger.error(f"Iteration {iteration} failed. Stopping.")
            break
        
        # If not the last iteration, prepare for the next one
        if iteration < total_iterations:
            # For subsequent iterations, we need just the "Humanized Text" column
            if not prepare_for_next_iteration(iteration_output_csv):
                logger.error("Failed to prepare for next iteration. Stopping.")
                break
            
            # Set the current iteration's output as the next iteration's input
            current_input_csv = iteration_output_csv
    
    logger.info("\nAll iterations completed!")

if __name__ == "__main__":
    main()
