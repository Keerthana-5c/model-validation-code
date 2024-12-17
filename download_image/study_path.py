import os
import pandas as pd
import requests
import csv
import concurrent.futures as futures
from tqdm import tqdm

def get_study_path(study_iuid):
    path = requests.get('https://api.5cnetwork.com/dicom/storage-path/' + study_iuid).json()
    return path.get('path')

def get_study_paths(input_csv, output_csv):
    studies = pd.read_csv(input_csv)
    studies["path"] = None

    def multi_thread(index, row):
        study_iuid = row["study_iuid"]
        try:
            path = get_study_path(study_iuid)
            studies.at[index, "path"] = path
        except Exception as e:
            print(f"Error processing study_iuid {study_iuid}: {e}")

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(studies.columns.tolist())

    with tqdm(total=len(studies)) as pbar:
        with futures.ThreadPoolExecutor(max_workers=64) as executors:
            to_do = []
            for i, row in studies.iterrows():
                future = executors.submit(multi_thread, i, row)
                future.add_done_callback(lambda p: pbar.update())
                to_do.append(future)
            futures.wait(to_do)

    studies.to_csv(output_csv, index=False)
    print(f'Obtained study paths for all the studies. Check {output_csv}!\n')
