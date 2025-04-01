import csv
import re

def extract_first_paragraph(input_file, output_file):
    """
    Reads a CSV file where text fields might contain multiple paragraphs,
    extracts only the first paragraph from each text field,
    and writes the result to a new CSV file.
    """
    try:
        # Read the input CSV file
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # Get the header row
            header = next(reader)
            
            # Read all rows
            rows = []
            for row in reader:
                # Process each column in the row
                processed_row = []
                for cell in row:
                    # Split the text by multiple line breaks to identify paragraphs
                    paragraphs = re.split(r'\n\s*\n', cell)
                    
                    # Take only the first paragraph if it exists
                    first_paragraph = paragraphs[0].strip() if paragraphs else ""
                    
                    # Remove any single line breaks within the first paragraph
                    first_paragraph = re.sub(r'\n', ' ', first_paragraph)
                    
                    processed_row.append(first_paragraph)
                rows.append(processed_row)
        
        # Write to the output CSV file
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Write the header
            writer.writerow(header)
            
            # Write the processed rows
            for row in rows:
                writer.writerow(row)
                
        print(f"Processing complete! Only first paragraphs have been extracted to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "initialText.csv"  # Replace with your input file name
    output_file = "newInitialText.csv"  # Output file name
    
    extract_first_paragraph(input_file, output_file)