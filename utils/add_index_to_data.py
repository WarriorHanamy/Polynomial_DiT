#!/usr/bin/env python3
"""
Add an index column to a raw data file.
"""

def add_index_to_data(input_file=None, output_file=None):
    """
    Add an index to each line of the data file.
    """
    print(f"Processing file: {input_file}")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    print(f"Number of original lines: {len(lines)}")
    
    with open(output_file, 'w') as f:
        for idx, line in enumerate(lines):
            indexed_line = f"{idx},{line.strip()}\n"
            f.write(indexed_line)
    
    print(f"Saved to: {output_file}")
    print("Format: index,original_data")
    
    print("\nFirst 3 lines preview:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.strip()}")
            else:
                break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Add index to each line in a data file.")
    parser.add_argument("--input_file", type=str, default=None, help="Input file path")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    args = parser.parse_args()
    add_index_to_data(input_file=args.input_file, output_file=args.output_file) 