import pandas as pd

def calculate_metrics():
    csv_path = 'csv/output.csv'  # Define your actual output CSV path
    df = pd.read_csv(csv_path)
    
    TP = ((df['ground truth'] == 1) & (df['prediction'] == 1)).sum()
    FP = ((df['ground truth'] == 0) & (df['prediction'] == 1)).sum()
    TN = ((df['ground truth'] == 0) & (df['prediction'] == 0)).sum()
    FN = ((df['ground truth'] == 1) & (df['prediction'] == 0)).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    print(f'TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

def main():
    calculate_metrics()

if __name__ == "__main__":
    main()
