import json
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

def read_json_data(file_path):
    """Read data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

def process_with_gemini(prompt, api_key_manager, max_retries=3):
    """Send the prompt to Gemini API and get the response."""
    retries = 0
    while retries <= max_retries:
        try:
            # Get the current API key and client
            client = api_key_manager.get_client()
            
            response = client.models.generate_content(
                model="gemini-1.5-pro", 
                contents=prompt
            )
            
            # Record successful request
            api_key_manager.record_request()
            
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            logger = logging.getLogger()
            
            # Check if this is a rate limit error
            if "429 RESOURCE_EXHAUSTED" in error_str and "quota" in error_str.lower():
                logger.warning(f"Rate limit error detected: {error_str[:100]}...")
                
                # Try switching to next key
                api_key_manager.try_next_key()
                
                # If we have more keys to try, retry immediately
                if api_key_manager.has_available_keys():
                    logger.info(f"Retrying with new API key (attempt {retries+1}/{max_retries})")
                    retries += 1
                    continue
                else:
                    logger.error("All API keys have reached their quota. Cannot proceed.")
                    return f"Error: All API keys exhausted. Last error: {str(e)[:100]}..."
            else:
                # For other errors, don't retry with a different key
                logger.error(f"API error (not rate-limited): {e}")
                return f"Error processing: {str(e)}"
    
    return f"Error: Failed after {max_retries} retries with different API keys"

def save_results_to_json(data, output_file):
    """Save results to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Results successfully saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving JSON results: {e}")
        return False

class ApiKeyManager:
    """Manager for rotating through multiple API keys."""
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.request_counts = [0] * len(api_keys)
        self.exhausted_keys = [False] * len(api_keys)  # Track which keys hit quota limits
        self.clients = [genai.Client(api_key=key) for key in api_keys]
        self.max_requests_per_key = 50
        self.logger = logging.getLogger()
        
    def get_client(self):
        """Get the current client."""
        return self.clients[self.current_key_index]
    
    def record_request(self):
        """Record a successful request with the current key."""
        self.request_counts[self.current_key_index] += 1
        
        # If we're close to the limit, switch to next key
        if self.request_counts[self.current_key_index] >= self.max_requests_per_key:
            self.switch_to_next_key()
    
    def try_next_key(self):
        """Try switching to the next key in case of errors."""
        # Mark current key as exhausted
        self.exhausted_keys[self.current_key_index] = True
        self.logger.warning(f"API Key {self.current_key_index+1} hit quota limit, marking as exhausted")
        self.switch_to_next_key()
    
    def switch_to_next_key(self):
        """Switch to the next available API key."""
        old_index = self.current_key_index
        
        # Find the next non-exhausted key
        for i in range(1, len(self.api_keys)):
            next_index = (old_index + i) % len(self.api_keys)
            if not self.exhausted_keys[next_index]:
                self.current_key_index = next_index
                self.logger.info(f"Switching from API key {old_index+1} (used {self.request_counts[old_index]} requests) to API key {self.current_key_index+1}")
                return
        
        # If we got here, all keys are marked as exhausted
        self.logger.warning("All API keys appear to be exhausted! Will try the least used key.")
        
        # Find the key with the lowest usage count
        min_usage = min(self.request_counts)
        min_indices = [i for i, count in enumerate(self.request_counts) if count == min_usage]
        if min_indices:
            self.current_key_index = min_indices[0]
            # Reset the exhausted flag for this key to give it another chance
            self.exhausted_keys[self.current_key_index] = False
            self.logger.info(f"Attempting to reuse least-used API key {self.current_key_index+1} (usage: {min_usage})")
    
    def has_available_keys(self):
        """Check if there are any non-exhausted API keys available."""
        return not all(self.exhausted_keys)
                
    def get_usage_report(self):
        """Get a report of API key usage."""
        return {
            f"API Key {i+1}": {
                "requests": count, 
                "exhausted": self.exhausted_keys[i]
            } 
            for i, count in enumerate(self.request_counts)
        }

def process_batch(texts_data, start_idx, end_idx, prompt_template, api_key_manager, iteration_num, logger):
    """Process a batch of texts."""
    batch_start_time = time.time()
    logger.info(f"Starting batch processing for texts {start_idx+1}-{end_idx} of {len(texts_data)}")
    
    # Process each text in current batch
    i = start_idx
    while i < end_idx and i < len(texts_data):
        text_item = texts_data[i]
        logger.info(f"Processing text {i+1}/{len(texts_data)}...")
        
        try:
            # Get the text to process based on iteration
            if iteration_num == 1:
                text_to_process = text_item["summarized_text"]
            else:
                # Use the latest iteration's text
                text_to_process = text_item["iterations"][-1]["rewritten_text"]
            
            # Prepare the prompt by replacing placeholder with the text
            prompt_start_time = time.time()
            full_prompt = prompt_template.replace("Here is the text:", f"Here is the text:\n{text_to_process}")
            prompt_prep_time = time.time() - prompt_start_time
            logger.debug(f"Prompt preparation took {prompt_prep_time:.2f} seconds")
            
            # Send to Gemini API
            api_start_time = time.time()
            rewritten_text = process_with_gemini(full_prompt, api_key_manager)
            api_time = time.time() - api_start_time
            logger.debug(f"API call took {api_time:.2f} seconds")
            
            # Check if we got an error response that indicates all keys are exhausted
            if rewritten_text.startswith("Error: All API keys exhausted"):
                logger.error("All API keys have reached their quota. Stopping batch processing.")
                return False
            
            # Add iteration data to the text item
            if "iterations" not in text_item:
                text_item["iterations"] = []
                
            text_item["iterations"].append({
                "iteration_number": iteration_num,
                "rewritten_text": rewritten_text
            })
            
            # Only increment the counter if we successfully processed this text
            i += 1
            
            # Add a small delay between individual requests
            if i < end_idx:
                time.sleep(2)
        except Exception as e:
            logger.error(f"Error processing text {i+1}: {e}")
            # If we hit a general error, try to continue with the next text
            i += 1
    
    batch_time = time.time() - batch_start_time
    logger.info(f"Batch processing complete. Took {batch_time:.2f} seconds")
    return True

def run_iteration(data, prompt_template, api_key_manager, iteration_num, logger):
    """Run a single iteration of the text processing."""
    logger.info(f"\n{'='*40}")
    logger.info(f"STARTING ITERATION {iteration_num}")
    logger.info(f"{'='*40}")
    
    if not data:
        logger.error("No data to process.")
        return False
    
    logger.info(f"Found {len(data)} texts to process.")
    
    # Process texts in batches
    batch_size = 2  # Changed from 15 to 2 as requested
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    try:
        for batch_num in range(total_batches):
            logger.info(f"\nProcessing batch {batch_num+1}/{total_batches}...")
            
            # Calculate batch start and end
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(data))
            
            # Process the batch
            success = process_batch(
                data, start_idx, end_idx, prompt_template, api_key_manager, 
                iteration_num, logger=logger
            )
            
            if not success:
                return False
            
            # Wait 60 seconds before the next batch (unless it's the last batch)
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

def main():
    # Initialize API keys
    api_keys = [
        "AIzaSyDjkDzWNORGy8-Da2YDQeaNC0hip2QLwCw",  # Original key
        "AIzaSyAzP5KTYdSECCFvYFE7RJCOGB_336PcmHw",  # New key 2
        "AIzaSyCiuqT6xEQmp2S8veKe-mhKHUq1VLOEVx8",  # New key 3
        "AIzaSyCSuTBJvpANHU_wQlZbIFgAoqXhtdfAFoc",  # New key 4
        "AIzaSyDvjTIAb6j2maLiqoyTREGrVzVruKOftVE",  # New key 5
        "AIzaSyBNKM03kukcWRMaqGJKGE8w74BPKSpt-DY",  # New key 6
        "AIzaSyDjkDzWNORGy8-Da2YDQeaNC0hip2QLwCw",  # New key 7
    ]
    
    # Create API key manager
    api_key_manager = ApiKeyManager(api_keys)
    
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
    logger.info(f"Initialized with {len(api_keys)} API keys")
    
    # Read the prompt template
    prompt_template = read_prompt_template(prompt_template_path)
    if not prompt_template:
        logger.error("Failed to load prompt template.")
        return
    
    # Read initial data
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
        success = run_iteration(
            data=data,
            prompt_template=prompt_template,
            api_key_manager=api_key_manager,
            iteration_num=iteration,
            logger=logger
        )
        
        if not success:
            logger.error(f"Iteration {iteration} failed. Stopping.")
            break
        
        # Save the interim output after each iteration
        interim_output = os.path.join(output_dir, f"iterations_output_step{iteration}.json")
        save_results_to_json(data, interim_output)
        logger.info(f"Saved interim output for iteration {iteration} to {interim_output}")
        
        # Log API key usage
        usage_report = api_key_manager.get_usage_report()
        logger.info(f"API Key Usage after iteration {iteration}:")
        for key, count in usage_report.items():
            logger.info(f"  {key}: {count} requests")
    
    # Save the final results
    save_results_to_json(data, output_json)
    logger.info(f"\nAll iterations completed! Final output saved to {output_json}")
    
    # Final API key usage report
    logger.info("\nFinal API Key Usage Report:")
    for key, count in api_key_manager.get_usage_report().items():
        logger.info(f"  {key}: {count} requests")

if __name__ == "__main__":
    main()
