import concurrent.futures
from utils.Dataset import utils as datasetUtils
import time, os
import pandas as pd

from_where_dict = {'Local': 0, 'Cloud': 0}

def setup_dataset_directories(dataset_root, dataset_limit):
    """
    Set up directories for storing dataset images and labels, each with a unique timestamp.
    
    Args:
    dataset_root (str): Base name for the dataset directory.
    """
    timestamp = str(int(time.time()))
    base_directory = f'{dataset_root}_{timestamp}'

    os.makedirs('data/', exist_ok=True)
    os.makedirs('data/csv', exist_ok=True)

    os.makedirs(f'data/{base_directory}/images', exist_ok=True)

    data_response = datasetUtils.read_data_from_postgres(dataset_root, column_type='findings', limit=dataset_limit)
    if data_response is not None:
        pd.DataFrame(data_response).to_csv(f'data/csv/{base_directory}.csv', index=False)
    else:
        raise ValueError("Failed to retrieve data from the database.")

    return base_directory

def process_image_classification(index, image_data, dataset_root, mode, print_interval, dataset_length):
    """
    Process and save classification data for a single image, and return the image path and label.
    
    Args:
    index (int): The index of the current image being processed.
    image_data (tuple): Contains the row data of the image.
    dataset_root (str): The root directory for the dataset.
    mode (str): Specifies the dataset type ('train' or 'test').
    print_interval (int): Interval at which to print progress.
    dataset_length (int): Total number of images in the dataset.
    
    Returns:
    tuple: A tuple containing the image file path and the label.
    """
    row = image_data
    row['image_path'] = row['image_path'].replace('.png', '.jpeg')
    image_file_path = f'data/{dataset_root}/images/{row["image_path"].replace("/", "_")}'


    if index % print_interval == 0:
        print(f"{mode} => {index} / {dataset_length}")

    if not os.path.exists(image_file_path):
        image, from_where = datasetUtils.get_image_from_source(row['image_path'])
        from_where_dict[from_where] += 1
        if not image:
            return 
        image.save(image_file_path)
    else:
        print('File already exists', image_file_path, mode)

    return {'image_path': image_file_path, 'label' : row['label'], 'mode' : mode}

def execute_classification_processing(dataset_root, dataset_limit='NULL', print_interval=100):
    """
    Orchestrates the processing of classification data for the entire dataset.
    
    Args:
    dataset_root (str): The root directory name for the dataset.
    print_interval (int): Interval at which to print progress during processing.
    """
    dataset_root = setup_dataset_directories(dataset_root, dataset_limit)
    dataset = datasetUtils.load_dataset_and_shuffle(dataset_root)
    train_dataset, test_dataset = datasetUtils.split_dataset(dataset, train_ratio=0.8)

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_classification, index, row, dataset_root, 'train', print_interval, len(train_dataset)) for index, row in train_dataset.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_classification, index, row, dataset_root, 'test', print_interval, len(test_dataset)) for index, row in test_dataset.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results = [result for result in results if result is not None]
    pd.DataFrame(results).to_csv(f'data/{dataset_root}/labels.csv', index = False)

    total_images = from_where_dict['Local'] + from_where_dict['Cloud']
    local_percentage = (from_where_dict['Local'] / total_images) * 100 if total_images > 0 else 0
    global_percentage = (from_where_dict['Cloud'] / total_images) * 100 if total_images > 0 else 0
    print(f"{local_percentage:.2f}% are downloaded from Local and {global_percentage:.2f}% images from Cloud")
   
    return dataset_root