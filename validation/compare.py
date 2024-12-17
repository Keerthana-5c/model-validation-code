import pandas as pd

def process_csv(mapping_column):
    csv1_path = '/home/ai-user/test_models/download_image/csv/download_status.csv'  # Define your actual CSV file path
    csv2_path = '/home/ai-user/test_models/validation/csv/prediction_results.csv'  # Define your actual CSV file path
    
    df1 = pd.read_csv(csv1_path)
    
    unique_names = set()

    # Extract and clean names from 'hil_pathology'
    for pathologies in df1['hil_pathology'].dropna():
        # cleaned_names = [name.strip().lower().replace(' ', '_').replace('foreign_body_-_', '') for name in pathologies.split(',')]

        cleaned_names = [
            # If the name starts with "foreign", rename it to "foreign_body"
            'foreign_body' if name.strip().lower().startswith('foreign body') else
            name.strip().lower().replace(' ', '_').replace('foreign_body_-_', '')
            for name in pathologies.split(',')
        ]
        
        unique_names.update(cleaned_names)

    # Create cleaned names dictionary
    cleaned_names_dict = {name: name for name in unique_names}

    # Add cleaned names as columns to df1
    for cleaned_name in unique_names:
        df1[cleaned_name] = df1['hil_pathology'].apply(
            lambda x: 1 if pd.notna(x) and cleaned_name in [
                n.strip().lower().replace(' ', '_').replace('foreign_body_-_', '')
                for n in x.split(',')
            ] else 0
        )

    # Add a 'foreign_body' column to mark entries starting with "foreign body"
    df1['foreign_body'] = df1['hil_pathology'].apply(
        lambda x: 1 if pd.notna(x) and any(name.strip().lower().startswith('foreign body') for name in x.split(',')) else 0
    )

    # df1.to_csv('comparative.csv', index=False)

    # Create ground truth map
    if mapping_column not in df1.columns:
        raise KeyError(f"Column '{mapping_column}' does not exist in the first CSV.")
    
    ground_truth_map = df1.set_index('path')[mapping_column].to_dict()
    ground_truth_map = {key.replace('/', '_'): value for key, value in ground_truth_map.items()}
  
    # Load the second CSV
    df2 = pd.read_csv(csv2_path)

    # Check if the necessary columns exist in df2
    if 'Image Name' not in df2.columns or 'Prediction' not in df2.columns:
        raise KeyError("'Image Name' or 'Prediction' column does not exist in the second CSV.")

    # Map the ground truth values to df2 based on the Image Name
    df2['mapped_path'] = df2['Image Name'].str.replace('.jpeg', '', regex=False).str.replace('/', '_', regex=False)
    df2['mapped_path'] = df2['mapped_path'].str.split('_', n=4).str[:4].str.join('_')
    
    df2['ground_truth'] = df2['mapped_path'].map(ground_truth_map)
   
    df2.to_csv('csv2.csv', index=False)

    # Save the output to a new CSV
    final_output = pd.DataFrame({
        'image name': df2['Image Name'],
        'ground truth': df2['ground_truth'],
        'prediction': df2['Prediction']
    })

    final_output.to_csv('csv/output.csv', index=False)
    print("Output saved to 'output.csv'.")

def main(mapping_column):
    process_csv(mapping_column)

if __name__ == "__main__":
    main('your_mapping_column')  # Define your actual mapping column
