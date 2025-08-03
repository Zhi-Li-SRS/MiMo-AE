"""
This script is used to generate the cleaned data from the raw data.
The raw data is in the data directory.
The cleaned data is in the cleaned_data directory.

Based on config.yaml, we extract:
- time: timestamp
- bp: blood pressure  
- breath_upper: breath upper rate
- ppg_fing: ppg finger data 
"""

import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def clean_data_file(input_file: str, output_file: str, target_columns: list):
    """
    Clean a single data file by extracting only the target columns.
    
    Args:
        input_file (str): Path to input raw data file
        output_file (str): Path to output cleaned data file  
        target_columns (list): List of column names to extract
    """
    print(f"Processing {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        print(f"  Original shape: {df.shape}")
        
        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            print(f"  Warning: Missing columns {missing_cols} in {input_file}")
            available_cols = [col for col in target_columns if col in df.columns]
        else:
            available_cols = target_columns
        
        cleaned_df = df[available_cols].copy()
        
        # Remove rows with any missing values
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.dropna()
        final_rows = len(cleaned_df)
        
        if initial_rows != final_rows:
            print(f"  Removed {initial_rows - final_rows} rows with missing values")
        
        print(f"  Cleaned shape: {cleaned_df.shape}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        cleaned_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")
        
    except Exception as e:
        print(f"  Error processing {input_file}: {str(e)}")


def main():
    """Main function to process all data files."""
    config = load_config()
    
    signal_cols = config['preprocessing']['signal_cols']
    target_columns = ['time'] + signal_cols
    
    print(f"Target columns: {target_columns}")
    
    # Define input and output directories
    input_dir = Path('data')
    output_dir = Path('cleaned_data')
    
    output_dir.mkdir(exist_ok=True)
    
    input_files = list(input_dir.glob('*.csv'))
    
    if not input_files:
        print("No CSV files found in data directory!")
        return
    
    # Process each file
    for input_file in input_files:
        output_file = output_dir / input_file.name
        clean_data_file(str(input_file), str(output_file), target_columns)
    
    print("\nData cleaning completed!")


if __name__ == "__main__":
    main()
