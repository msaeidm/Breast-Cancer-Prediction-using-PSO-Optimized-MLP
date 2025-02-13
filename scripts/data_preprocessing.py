import pandas as pd
import os

def load_and_preprocess_data(file_path):
    # Load the raw data
    data = pd.read_csv(file_path)
    # Impute missing values with maximum frequency (mode) for each column
    for col in data.columns:
        mode_val = data[col].mode()[0]
        data[col].fillna(mode_val, inplace=True)
    # Normalize features using Min/Max scaling to [0, 1]
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
    return data

def main():
    input_file = os.path.join("..", "data", "raw", "breast_cancer_data.csv")
    processed_data = load_and_preprocess_data(input_file)
    os.makedirs(os.path.join("..", "data", "processed"), exist_ok=True)
    processed_file = os.path.join("..", "data", "processed", "processed_data.csv")
    processed_data.to_csv(processed_file, index=False)
    print(f"Processed data saved to {processed_file}")

if __name__ == "__main__":
    main()
