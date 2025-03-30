import csv
import os
from google import genai
import pandas as pd
import time

def read_csv_column(file_path, column_index=0, max_rows=None):
    """Read rows from a column in a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Ensure column index is valid
        if column_index >= len(df.columns):
            print(f"Column index {column_index} is out of range. Using first column.")
            column_index = 0
        
        # Get the column data (optionally limited by max_rows)
        if max_rows:
            texts = df.iloc[:max_rows, column_index].tolist()
        else:
            texts = df.iloc[:, column_index].tolist()
        return texts
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []

def rewrite_text(text, client):
    """Use Gemini API to rewrite the text."""
    prompt = f"""
Rewrite the following text in a new way while keeping the meaning, tone, and key details intact. 
Do not simply paraphrase; instead, restructure sentences, use varied phrasing, and make it feel 
naturally rewritten while keeping it recognizable as the same content. Ensure clarity and coherence. 
It should be around the similar length to the original given text below by the way:

{text}
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return "Error processing this text"

def save_results(results, output_file, append=False):
    """Save current results to CSV file."""
    mode = 'a' if append else 'w'
    with open(output_file, mode, newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(["Original Text", "Rewritten Text"])
        writer.writerows(results)
    print(f"Results saved to {output_file}")

def main():
    # Initialize Gemini client
    api_key = "AIzaSyBNKM03kukcWRMaqGJKGE8w74BPKSpt-DY"
    client = genai.Client(api_key=api_key)
    
    # File paths
    input_csv = input("Enter path to CSV file: ")
    output_csv = input("Enter path for output CSV file: ")
    
    # Read all texts from CSV
    column_index = 0  # Column 1 (0-indexed)
    max_rows = 150    # Process up to 150 rows (10 batches of 15)
    texts = read_csv_column(input_csv, column_index, max_rows)
    
    if not texts:
        print("No texts found or error reading file.")
        return
    
    # Process texts in batches
    batch_size = 15
    total_batches = 10 if len(texts) > 10 * batch_size else (len(texts) + batch_size - 1) // batch_size
    
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
                    save_results(all_results, output_csv, batch_num > 0)
                    return
            
            # Batch complete - save results
            save_results(batch_results, output_csv, batch_num > 0)
            
            # Wait a minute before the next batch (unless it's the last batch)
            if batch_num < total_batches - 1:
                print(f"Batch {batch_num+1} complete. Waiting 60 seconds before next batch...")
                time.sleep(60)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    finally:
        # Save any remaining results if they haven't been saved yet
        if all_results:
            print("Saving all collected results...")
            save_results(all_results, output_csv.replace('.csv', '_complete.csv'))
    
    print("Process completed!")

if __name__ == "__main__":
    main()
