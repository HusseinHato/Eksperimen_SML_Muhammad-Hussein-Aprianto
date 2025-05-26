import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from typing import Tuple
import os

def preprocess_crop_data(
    input_path: str, 
    output_path: str,
    numerical_cols: list = ['K', 'N', 'P', 'temperature', 'humidity', 'ph', 'rainfall'],
    iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocess crop recommendation data by handling missing values, duplicates, outliers, and scaling.
    
    Args:
        input_path (str): Path to the raw CSV file
        output_path (str): Path to save the preprocessed CSV file
        numerical_cols (list): List of numerical column names to process
        iqr_multiplier (float): Multiplier for IQR to detect outliers
        
    Returns:
        Tuple containing:
            - pd.DataFrame: Preprocessed DataFrame
            - dict: Dictionary containing preprocessing information
    """
    # Read and clean data
    try:
        df = pd.read_csv(input_path)
        initial_rows = len(df)
        
        # Basic cleaning
        df = df.dropna().drop_duplicates()
        cleaned_rows = len(df)
        
        # Handle outliers
        outliers_removed = {}
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Count and remove outliers
            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            outliers_removed[col] = len(df) - mask.sum()
            df = df[mask]
        
        # Scale numerical features
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Encode labels
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        
        # Save processed data
        df.to_csv(output_path, index=False)
        
        # Prepare summary
        summary = {
            'initial_rows': initial_rows,
            'rows_after_cleaning': cleaned_rows,
            'final_rows': len(df),
            'outliers_removed': outliers_removed,
            'label_classes': dict(enumerate(label_encoder.classes_))
        }
        
        return df, summary
        
    except Exception as e:
        raise Exception(f"Error during preprocessing: {str(e)}")

# Example usage
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    input_path = os.path.join(root_dir, 'croprecommendation_raw.csv')
    output_path = os.path.join(script_dir, 'croprecommendation_preprocessing.csv')
    
    try:
        processed_df, summary = preprocess_crop_data(input_path, output_path)
        print("Preprocessing completed successfully!")
        print(f"Initial rows: {summary['initial_rows']}")
        print(f"Rows after cleaning: {summary['rows_after_cleaning']}")
        print(f"Final rows after outlier removal: {summary['final_rows']}")
        print("Outliers removed by column:", summary['outliers_removed'])
        print("Label classes:", summary['label_classes'])
    except Exception as e:
        print(f"Error: {str(e)}")
