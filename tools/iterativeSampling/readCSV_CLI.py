import pandas as pd
import csv
import os
import sys
import argparse
from datetime import datetime

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Large CSV File Reader - Process and analyze large CSV files efficiently')
    
    # File path argument
    parser.add_argument('--file', '-f', default="AI_Human.csv", 
                        help='Path to the CSV file (default: AI_Human.csv)')
    
    # Display options
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show basic file information')
    parser.add_argument('--rows', '-r', type=str,
                        help='Show specific rows (format: start:end or single row number)')
    parser.add_argument('--columns', '-c', type=str,
                        help='Show only specific columns (comma-separated column names)')
    parser.add_argument('--sample', '-s', type=int, default=10,
                        help='Show a random sample of N rows (default: 10)')
    parser.add_argument('--no-truncate', '-nt', action='store_true',
                        help='Prevent pandas from truncating data in display')
    
    # Analysis options
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='Perform basic analysis on the file')
    parser.add_argument('--search', type=str, nargs=2, metavar=('COLUMN', 'VALUE'),
                        help='Search for VALUE in COLUMN')
    parser.add_argument('--process', '-p', action='store_true',
                        help='Process the entire file')
    parser.add_argument('--chunk-size', type=int, default=100000,
                        help='Chunk size for processing (default: 100000)')
    parser.add_argument('--output-dir', default='processed_data',
                        help='Output directory for processed data (default: processed_data)')
    
    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                        help='Launch interactive exploration mode')
    
    return parser.parse_args()

def get_file_info(file_path):
    """Get basic information about the CSV file."""
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    
    print(f"File: {file_path}")
    print(f"Size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    print(f"Last modified: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
    
    # Get column names and count rows
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # Count a sample of rows to estimate total
        sample_size = 1000
        sample_rows = []
        for i, row in enumerate(reader):
            if i < sample_size:
                sample_rows.append(row)
            else:
                break
        
        # Estimate row count if file is large
        if file_size_bytes > 100 * 1024 * 1024:  # If > 100MB
            avg_row_size = file_size_bytes / (len(sample_rows) + 1)  # +1 for header
            estimated_rows = int(file_size_bytes / avg_row_size)
            print(f"Estimated rows: ~{estimated_rows:,} (based on sample)")
        else:
            # Count exact rows for smaller files
            row_count = len(sample_rows) + sum(1 for _ in reader)
            print(f"Row count: {row_count:,}")
    
    print(f"Columns ({len(headers)}):")
    for col in headers:
        print(f"  - {col}")
    
    return headers, sample_rows

def chunk_reader(file_path, chunk_size=100000):
    """Generator function to read the CSV in chunks."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield chunk

def get_specific_rows(file_path, row_spec, columns=None):
    """Get specific rows from the CSV file.
    
    Args:
        file_path: Path to the CSV file
        row_spec: String in the format "start:end" or a single row number
        columns: List of column names to include, or None for all columns
    """
    print(f"\nRetrieving rows: {row_spec}")
    
    # Parse row specification
    if ':' in row_spec:
        start, end = map(int, row_spec.split(':'))
        row_indices = range(start, end + 1)
    else:
        try:
            row_idx = int(row_spec)
            row_indices = [row_idx]
        except ValueError:
            print(f"Invalid row specification: {row_spec}. Use format 'start:end' or a single number.")
            return
    
    # Calculate the maximum row index we need
    max_row_idx = max(row_indices)
    
    # Read the file with specific columns if provided
    if columns:
        # First get all column names to validate the provided columns exist
        with open(file_path, 'r', encoding='utf-8') as f:
            all_columns = next(csv.reader(f))
        
        # Validate columns
        valid_columns = [col for col in columns if col in all_columns]
        if len(valid_columns) < len(columns):
            invalid_cols = set(columns) - set(valid_columns)
            print(f"Warning: These columns do not exist: {', '.join(invalid_cols)}")
            
        if not valid_columns:
            print("No valid columns specified. Showing all columns.")
            usecols = None
        else:
            usecols = valid_columns
    else:
        usecols = None
    
    # Determine chunk size based on max row index
    chunk_size = max(10000, max_row_idx // 10 + 1)
    
    # Read the file in chunks to find the rows
    current_row = 0
    requested_rows = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, usecols=usecols):
        chunk_end_row = current_row + len(chunk)
        
        # Extract rows that fall within this chunk
        for idx in row_indices:
            if current_row <= idx < chunk_end_row:
                # Adjust index for the chunk
                chunk_row_idx = idx - current_row
                if 0 <= chunk_row_idx < len(chunk):
                    requested_rows.append(chunk.iloc[chunk_row_idx])
        
        current_row = chunk_end_row
        
        # If we've passed the highest requested row index, we can stop
        if current_row > max_row_idx:
            break
    
    if requested_rows:
        return pd.DataFrame(requested_rows)
    else:
        print(f"No rows found with indices {row_spec}")
        return None

def analyze_file(file_path, chunk_size=100000):
    """Perform basic analysis on the file."""
    print("\nAnalyzing file content...")
    headers, _ = get_file_info(file_path)
    
    # Process the first few chunks to get a sense of the data
    chunks_to_analyze = 3
    total_rows_analyzed = 0
    
    # Data type inference
    sample_data = {}
    
    for i, chunk in enumerate(chunk_reader(file_path, chunk_size)):
        if i >= chunks_to_analyze:
            break
            
        total_rows_analyzed += len(chunk)
        
        # Get data types and sample values for each column
        for col in headers:
            if col in chunk.columns:
                if col not in sample_data:
                    sample_data[col] = {
                        'dtype': str(chunk[col].dtype),
                        'non_null': chunk[col].count(),
                        'null_count': chunk[col].isna().sum(),
                        'unique_values': min(5, chunk[col].nunique()),
                        'sample_values': chunk[col].dropna().sample(min(5, chunk[col].count())).tolist()
                    }
                else:
                    sample_data[col]['non_null'] += chunk[col].count()
                    sample_data[col]['null_count'] += chunk[col].isna().sum()
    
    print(f"\nAnalyzed {total_rows_analyzed:,} rows from first {chunks_to_analyze} chunks")
    
    print("\nColumn Analysis:")
    for col, data in sample_data.items():
        null_percent = data['null_count'] / (data['non_null'] + data['null_count']) * 100 if (data['non_null'] + data['null_count']) > 0 else 0
        print(f"\n{col}:")
        print(f"  Data type: {data['dtype']}")
        print(f"  Null values: {data['null_count']:,} ({null_percent:.1f}%)")
        print(f"  Sample values: {data['sample_values']}")

def search_in_file(file_path, column, value, chunk_size=100000):
    """Search for a specific value in a column."""
    print(f"\nSearching for '{value}' in column '{column}'...")
    
    # First check if column exists
    with open(file_path, 'r', encoding='utf-8') as f:
        headers = next(csv.reader(f))
    
    if column not in headers:
        print(f"Error: Column '{column}' not found in the file.")
        print(f"Available columns: {', '.join(headers)}")
        return
    
    found_count = 0
    found_rows = []
    
    for i, chunk in enumerate(chunk_reader(file_path, chunk_size)):
        if column in chunk.columns:
            # String search for all data types
            matches = chunk[chunk[column].astype(str).str.contains(str(value), na=False)]
            
            if not matches.empty:
                found_count += len(matches)
                found_rows.append(matches)
                
                if len(found_rows) == 1 and i == 0:
                    print(f"\nFirst {min(5, len(matches))} matches:")
                    print(matches.head(5))
        
        # Give progress updates
        if i % 10 == 0 and i > 0:
            print(f"Searched {i} chunks... Found {found_count} matches so far.")
    
    print(f"\nSearch complete. Found {found_count} total matches.")
    
    if found_count > 0:
        result = pd.concat(found_rows)
        return result.head(50)  # Return first 50 matches
    return None

def process_file(file_path, output_dir='processed_data', chunk_size=100000):
    """Process the entire file in chunks and save results."""
    os.makedirs(output_dir, exist_ok=True)
    
    total_chunks = 0
    total_rows = 0
    
    start_time = datetime.now()
    print(f"\nStarting processing at {start_time}")
    
    for i, chunk in enumerate(chunk_reader(file_path, chunk_size)):
        # Example processing: save each chunk as a separate file
        # Modify this section to perform your specific processing
        chunk_file = os.path.join(output_dir, f"chunk_{i:04d}.csv")
        chunk.to_csv(chunk_file, index=False)
        
        total_chunks += 1
        total_rows += len(chunk)
        
        # Progress update every 5 chunks
        if i % 5 == 0:
            elapsed = datetime.now() - start_time
            print(f"Processed {total_rows:,} rows in {total_chunks} chunks. Elapsed time: {elapsed}")
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    print(f"\nProcessing complete!")
    print(f"Total chunks: {total_chunks}")
    print(f"Total rows: {total_rows:,}")
    print(f"Processing time: {processing_time}")
    print(f"Average processing speed: {total_rows / processing_time.total_seconds():,.1f} rows/second")

def get_random_sample(file_path, sample_size=10, columns=None):
    """Get a random sample of rows from the file."""
    # First get file size to estimate total rows
    file_size = os.path.getsize(file_path)
    
    # Open the file to get headers and estimate rows
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        # Count the first 1000 rows to estimate average row size
        rows_to_count = 1000
        for i in range(rows_to_count):
            try:
                next(reader)
            except StopIteration:
                rows_to_count = i
                break
    
    # Estimate total rows based on average row size
    avg_row_size = file_size / (rows_to_count + 1)  # +1 for header
    estimated_rows = int(file_size / avg_row_size)
    
    # Generate random row numbers to extract
    import random
    random.seed(datetime.now().timestamp())
    if estimated_rows <= sample_size:
        random_rows = list(range(estimated_rows))
    else:
        random_rows = sorted(random.sample(range(estimated_rows), sample_size))
    
    # Read specific rows
    samples = []
    current_row = 0
    
    # Determine which columns to read
    if columns:
        valid_columns = [col for col in columns if col in headers]
        if len(valid_columns) < len(columns):
            invalid_cols = set(columns) - set(valid_columns)
            print(f"Warning: These columns do not exist: {', '.join(invalid_cols)}")
            
        if not valid_columns:
            print("No valid columns specified. Showing all columns.")
            usecols = None
        else:
            usecols = valid_columns
    else:
        usecols = None
    
    # Read in chunks to get the random rows
    for chunk in pd.read_csv(file_path, chunksize=10000, usecols=usecols):
        chunk_end_row = current_row + len(chunk)
        
        # Extract rows that fall within this chunk
        chunk_rows = []
        for row_idx in random_rows:
            if current_row <= row_idx < chunk_end_row:
                chunk_row = row_idx - current_row
                chunk_rows.append(chunk_row)
        
        if chunk_rows:
            samples.append(chunk.iloc[chunk_rows])
            
            # Remove the rows we've already found
            random_rows = [r for r in random_rows if r >= chunk_end_row]
            
            # If we've found all random rows, we can stop
            if not random_rows:
                break
        
        current_row = chunk_end_row
    
    if samples:
        return pd.concat(samples)
    else:
        print("No samples could be retrieved.")
        return None

def interactive_explore(file_path):
    """Interactively explore the CSV file."""
    headers, sample_rows = get_file_info(file_path)
    
    # Set pandas display options for the interactive session
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    while True:
        print("\nExplore options:")
        print("1. View first rows")
        print("2. View specific rows")
        print("3. View specific columns")
        print("4. View random sample")
        print("5. Basic file analysis")
        print("6. Search for values")
        print("7. Process entire file")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            num_rows = int(input("How many rows to display (default 10): ") or 10)
            first_chunk = next(chunk_reader(file_path, num_rows))
            print("\nFirst rows of data:")
            print(first_chunk.head(num_rows))
            
        elif choice == '2':
            row_spec = input("Enter row numbers to view (e.g., '0:5' for rows 0-5, or '42' for just row 42): ")
            result = get_specific_rows(file_path, row_spec)
            if result is not None:
                print("\nRequested rows:")
                print(result)
            
        elif choice == '3':
            cols_input = input(f"Enter column names to view (comma-separated): ")
            columns = [col.strip() for col in cols_input.split(',')]
            
            rows_input = input("Enter row range (default first 10 rows, e.g., '0:9'): ") or "0:9"
            result = get_specific_rows(file_path, rows_input, columns)
            
            if result is not None:
                print("\nRequested columns and rows:")
                print(result)
            
        elif choice == '4':
            sample_size = int(input("Enter sample size (default 10): ") or 10)
            cols_input = input("Enter specific columns to include (comma-separated, or leave empty for all): ")
            
            columns = None
            if cols_input:
                columns = [col.strip() for col in cols_input.split(',')]
                
            sample = get_random_sample(file_path, sample_size, columns)
            if sample is not None:
                print(f"\nRandom sample of {len(sample)} rows:")
                print(sample)
            
        elif choice == '5':
            analyze_file(file_path)
            
        elif choice == '6':
            column = input(f"Enter column name to search: ")
            search_value = input("Enter value to search for: ")
            result = search_in_file(file_path, column, search_value)
            
            if result is not None and not result.empty:
                print("\nSearch results (showing up to 50 matches):")
                print(result)
            
        elif choice == '7':
            chunk_size = int(input("Enter chunk size (default 100000): ") or 100000)
            output_dir = input("Enter output directory (default 'processed_data'): ") or 'processed_data'
            process_file(file_path, output_dir, chunk_size)
            
        elif choice == '8':
            print("Exiting exploration.")
            break
            
        else:
            print("Invalid choice. Please enter a number from 1-8.")

def main():
    """Main function to handle command-line arguments and execute tasks."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set display options if no-truncate is enabled
    if args.no_truncate:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found.")
        sys.exit(1)
    
    print(f"\n=== Large CSV File Reader ===\n")
    
    # Default behavior if no specific options are provided
    if not any([args.info, args.rows, args.analyze, args.search, 
                args.process, args.interactive, args.sample, args.columns]):
        args.interactive = True
    
    # File info
    if args.info:
        get_file_info(args.file)
    
    # Analyze file
    if args.analyze:
        analyze_file(args.file, args.chunk_size)
    
    # Show specific rows with optional column filtering
    if args.rows:
        columns = args.columns.split(',') if args.columns else None
        result = get_specific_rows(args.file, args.rows, columns)
        if result is not None:
            print("\nRequested rows:")
            print(result)
    
    # Show only specific columns without row specification
    elif args.columns and not args.rows:
        columns = args.columns.split(',')
        # Default to first 10 rows
        result = get_specific_rows(args.file, "0:9", columns)
        if result is not None:
            print("\nFirst 10 rows with selected columns:")
            print(result)
    
    # Show random sample
    if args.sample and not args.rows:
        columns = args.columns.split(',') if args.columns else None
        sample = get_random_sample(args.file, args.sample, columns)
        if sample is not None:
            print(f"\nRandom sample of {args.sample} rows:")
            print(sample)
    
    # Search for values
    if args.search:
        column, value = args.search
        search_in_file(args.file, column, value, args.chunk_size)
    
    # Process file
    if args.process:
        process_file(args.file, args.output_dir, args.chunk_size)
    
    # Interactive mode
    if args.interactive:
        interactive_explore(args.file)

if __name__ == "__main__":
    main()