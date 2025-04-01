import json
import os
import time
import pandas as pd
import openai
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

def read_json_data(file_path):
    """Read the JSON data file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def process_with_gpt4o(prompt, client):
    """Send the prompt to GPT-4o API and get the response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with GPT-4o API: {e}")
        return f"Error processing: {str(e)}"

def save_results(results, output_file):
    """Save results to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Results successfully saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def process_batch(texts, start_idx, end_idx, prompt_template, client, iteration_num, logger):
    """Process a batch of texts."""
    batch_start_time = time.time()
    logger.info(f"Starting batch processing for texts {start_idx+1}-{end_idx} of {len(texts)}")
    
    batch_results = []
    # Process each text in current batch
    for i in range(start_idx, end_idx):
        text_obj = texts[i]
        logger.info(f"Processing text {i+1}/{len(texts)} (ID: {text_obj['id']})...")
        
        try:
            # Determine which text to use based on the iteration
            if iteration_num == 1:
                input_text = text_obj["summarized_text"]
            else:
                # Get the text from previous iteration
                input_text = text_obj["iterations"][-1]["rewritten_text"]
            
            # Prepare the prompt by replacing placeholder with the text
            prompt_start_time = time.time()
            full_prompt = prompt_template.replace("Here is the text:", f"Here is the text:\n{input_text}")
            prompt_prep_time = time.time() - prompt_start_time
            logger.debug(f"Prompt preparation took {prompt_prep_time:.2f} seconds")
            
            # Send to GPT-4o API
            api_start_time = time.time()
            rewritten_text = process_with_gpt4o(full_prompt, client)
            api_time = time.time() - api_start_time
            logger.debug(f"API call took {api_time:.2f} seconds")
            
            # Add this iteration result to the text object
            iteration_result = {
                "iteration_number": iteration_num,
                "rewritten_text": rewritten_text
            }
            
            if "iterations" not in text_obj:
                text_obj["iterations"] = []
                
            text_obj["iterations"].append(iteration_result)
            batch_results.append(text_obj)
            
            # Add a small delay between individual requests
            if i < end_idx - 1:
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
            return False, texts
    
    batch_time = time.time() - batch_start_time
    logger.info(f"Batch processing complete. Took {batch_time:.2f} seconds")
    return True, texts

def run_iteration(data, prompt_template, client, iteration_num, logger):
    """Run a single iteration of the text processing."""
    logger.info(f"\n{'='*40}")
    logger.info(f"STARTING ITERATION {iteration_num}")
    logger.info(f"{'='*40}")
    
    if not data:
        logger.error("No texts found in the JSON data.")
        return False, data
    
    logger.info(f"Found {len(data)} texts to process.")
    
    # Process texts in batches
    batch_size = 15
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    try:
        for batch_num in range(total_batches):
            logger.info(f"\nProcessing batch {batch_num+1}/{total_batches}...")
            
            # Calculate batch start and end
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            
            # Process the batch
            success, data = process_batch(
                data, start_idx, end_idx, prompt_template, client, iteration_num, logger
            )
            
            if not success:
                return False, data
            
            # Wait a minute before the next batch (unless it's the last batch)
            if batch_num < total_batches - 1:
                logger.info(f"Batch {batch_num+1} complete. Waiting 60 seconds before next batch...")
                time.sleep(60)
    
    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user.")
        return False, data
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        return False, data
    
    logger.info(f"Iteration {iteration_num} completed successfully!")
    return True, data

def main():
    # Initialize OpenAI API client
    api_key = input("Enter your OpenAI API key: ")
    client = openai.OpenAI(api_key=api_key)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_template_path = os.path.join(script_dir, "humanizationPrompt.txt")
    
    # Get initial input
    initial_input_json = input("Enter path to initial input JSON file: ")
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
    
    # Read the initial JSON data
    data = read_json_data(initial_input_json)
    if not data:
        logger.error("Failed to load input JSON data.")
        return
    
    # Run 10 iterations
    total_iterations = 10
    output_json = os.path.join(output_dir, "iterations_output.json")
    
    for iteration in range(1, total_iterations + 1):
        logger.info(f"\nStarting iteration {iteration}/{total_iterations}")
        
        # Run this iteration
        success, data = run_iteration(
            data=data,
            prompt_template=prompt_template,
            client=client,
            iteration_num=iteration,
            logger=logger
        )
        
        if not success:
            logger.error(f"Iteration {iteration} failed. Stopping.")
            break
        
        # Save the current state after each iteration
        interim_output = os.path.join(output_dir, f"iterations_output_step{iteration}.json")
        save_results(data, interim_output)
    
    # Save the final results
    save_results(data, output_json)
    logger.info(f"\nAll iterations completed! Final output saved to {output_json}")

if __name__ == "__main__":
    main()
