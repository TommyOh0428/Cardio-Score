import pandas as pd
import os
from src.config import DATA_DIR

def extract_features(input_path, output_path):
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # Columns to keep
    columns_to_keep = [
        'Sex',
        'GeneralHealth',
        'PhysicalActivities',
        'SleepHours',
        'HadHeartAttack',
        'SmokerStatus',
        'RaceEthnicityCategory',
        'AgeCategory', # Corrected spelling from user request 'AgeCatagory'
        'HeightInMeters',
        'WeightInKilograms',
        'AlcoholDrinkers'
    ]

    try:
        # Load the dataset
        print(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path)
        
        # Verify columns exist (handle potential spelling mismatches)
        available_columns = df.columns.tolist()
        missing_cols = [col for col in columns_to_keep if col not in available_columns]
        
        if missing_cols:
            print(f"Warning: The following columns were not found in the dataset: {missing_cols}")
            print("Please check output file to see what was saved.")
            # Only keep columns that actually exist
            columns_to_keep = [col for col in columns_to_keep if col in available_columns]

        if not columns_to_keep:
            print("Error: None of the requested columns were found!")
            return

        # Create new dataframe with selected columns
        df_selected = df[columns_to_keep]

        # Save to new CSV
        df_selected.to_csv(output_path, index=False)
        print(f"✅ Successfully created new dataset at: {output_path}")
        print(f"Original Row Count: {len(df)}")
        print(f"Selected Columns: {len(columns_to_keep)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can change these filenames as needed
    # Updated to match the existing file in data/raw
    INPUT_FILE = os.path.join(DATA_DIR, 'raw/heart_2022_with_nans.csv') 
    OUTPUT_FILE = os.path.join(DATA_DIR, 'heart_selected_features.csv')
    extract_features(INPUT_FILE, OUTPUT_FILE)
