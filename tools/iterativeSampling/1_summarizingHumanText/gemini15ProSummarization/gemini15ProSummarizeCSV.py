import csv
import os
import json
import re
import random
from google import genai
import pandas as pd
import time

def extract_first_paragraph(text):
    """Extract only the first paragraph from the text.
    If the first paragraph is too short, include another paragraph if available."""
    if not text or not isinstance(text, str):
        return ""
    
    # Split by paragraphs (looking for double newlines or single newlines)
    paragraphs = text.split('\n\n')
    if len(paragraphs) == 1:
        paragraphs = text.split('\n')
    
    # If no clear paragraphs, return the whole text
    if len(paragraphs) == 1:
        return paragraphs[0].strip()
    
    # Check if first paragraph is too short (less than 100 characters)
    first_para = paragraphs[0].strip()
    if len(first_para) < 100 and len(paragraphs) > 1:
        # Include the second paragraph as well
        return '\n\n'.join([first_para, paragraphs[1].strip()])
    
    return first_para

def read_csv_column(file_path, column_index=0, max_rows=None):
    """Read rows from a column in a CSV file and extract first paragraphs."""
    try:
        df = pd.read_csv(file_path)
        # Ensure column index is valid
        if column_index >= len(df.columns):
            print(f"Column index {column_index} is out of range. Using first column.")
            column_index = 0
        
        # Get the column data (optionally limited by random max_rows)
        if max_rows and max_rows < len(df):
            # Get random sample instead of first N rows
            random_indices = random.sample(range(len(df)), max_rows)
            raw_texts = df.iloc[random_indices, column_index].tolist()
            print(f"Randomly selected {max_rows} rows from {len(df)} total rows")
        else:
            raw_texts = df.iloc[:, column_index].tolist()
            
        # Extract first paragraphs
        texts = [extract_first_paragraph(text) for text in raw_texts]
        return texts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def extract_retry_delay(error_message):
    """Extract retry delay from error message."""
    try:
        # Try to find the retry delay in the error message
        match = re.search(r"'retryDelay': '(\d+)s'", str(error_message))
        if match:
            return int(match.group(1)) + 2  # Add 2 seconds buffer to the retry delay
        return 62  # Default delay (60s) + 2s buffer if we can't parse the message
    except:
        return 62  # Default delay (60s) + 2s buffer on any parsing error

def rewrite_text(text, client, max_retries=3):
    """Use Gemini API to rewrite the text with retry functionality."""
    prompt = f"""
Rewrite the following text in a new way while keeping the meaning, tone, and key details intact. Restructure sentences, and use varied phrasing as necessary to make it feel naturally rewritten while keeping it recognizable as the same content. Ensure clarity and coherence. DO NOT ADD TITLES, LABELS, HEADINGS, OR WORDS OTHER THAN REWRITING THE TEXT. This rewrite should be around the similar length to the original given text which is given below:

{text}
"""
    
    retry_count = 0
    while retry_count <= max_retries:
        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro", 
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            print(f"Error with Gemini API: {e}")
            
            # Check if this is a rate limit error
            if "RESOURCE_EXHAUSTED" in error_str:
                # Extract retry delay from error message (now with +2s buffer added in the extraction function)
                retry_delay = extract_retry_delay(error_str)
                
                # Inform the user
                retry_count += 1
                if retry_count <= max_retries:
                    print(f"Rate limit exceeded. Retrying in {retry_delay} seconds (includes 2s buffer)... (Attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    print(f"Maximum retries ({max_retries}) exceeded. Skipping this text.")
                    return f"Error: Maximum retries exceeded after rate limit errors."
            else:
                # For other types of errors, don't retry
                return f"Error processing this text: {e}"
    
    return "Error: Failed after maximum retries"

def save_results(results, output_file, append=False):
    """Save current results to JSON file with the future-compatible structure."""
    data = []
    
    # If appending and file exists, load existing data first
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            # If file exists but is not valid JSON, start fresh
            data = []
    
    # If we're not appending, or no data was loaded, create new entries
    if not append or not data:
        # Add new results to data with IDs
        for i, (original, rewritten) in enumerate(results):
            data.append({
                "id": i + 1,
                "original_text": original,
                "summarized_text": rewritten,
                # Other fields will be added by future programs
            })
    else:
        # If we're appending, find the highest existing ID
        max_id = max(item.get("id", 0) for item in data)
        
        # Add new results to data with continuing IDs
        for i, (original, rewritten) in enumerate(results):
            data.append({
                "id": max_id + i + 1,
                "original_text": original,
                "summarized_text": rewritten,
                # Other fields will be added by future programs
            })
    
    # Write the combined data to file with pretty formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

def main():
    # Initialize Gemini client
    api_key = "AIzaSyBNKM03kukcWRMaqGJKGE8w74BPKSpt-DY"
    client = genai.Client(api_key=api_key)
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    # File paths
    input_csv = input("Enter path to CSV file: ")
    output_file = input("Enter path for output JSON file: ")
    
    # Ensure output file has .json extension
    if not output_file.lower().endswith('.json'):
        output_file += '.json'
    
    # Read random texts from CSV
    column_index = 0  # Column 1 (0-indexed)
    max_rows = 35     # Process 35 random rows
    texts = read_csv_column(input_csv, column_index, max_rows)
    
    if not texts:
        print("No texts found or error reading file.")
        return
    
    # Process texts in batches
    batch_size = 2  # Changed from 7 to 2
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    all_results = []
    
    try:
        for batch_num in range(total_batches):
            print(f"\nProcessing batch {batch_num+1}/{total_batches}...")
            
            # Calculate batch start and end
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            
            batch_results = []
            # Process each text in current batch
            for i, text in enumerate(texts[start_idx:end_idx]):
                current_idx = start_idx + i
                print(f"Processing text {current_idx+1}/{len(texts)}...")
                try:
                    rewritten = rewrite_text(text, client)
                    batch_results.append((text, rewritten))
                    all_results.append((text, rewritten))
                    # Small delay between individual requests
                    time.sleep(1)
                except Exception as e:
                    print(f"Error processing text {current_idx+1}: {e}")
                    # Save what we have so far if there's an error
                    save_results(all_results, output_file, batch_num > 0)
                    return
            
            # Batch complete - save results
            save_results(batch_results, output_file, batch_num > 0)
            
            # Wait 60 seconds between batches to avoid rate limiting
            if batch_num < total_batches - 1:  # Don't wait after the last batch
                print(f"Batch {batch_num+1} complete. Waiting 60 seconds before next batch...")
                time.sleep(60)
            else:
                print(f"Batch {batch_num+1} complete.")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Save any remaining results if they haven't been saved yet
        if all_results:
            print("Saving all collected results...")
            save_results(all_results, output_file.replace('.json', '_complete.json'))
    
    print("Process completed!")

if __name__ == "__main__":
    main()
