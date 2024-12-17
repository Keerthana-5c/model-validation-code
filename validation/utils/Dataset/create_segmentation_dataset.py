import os
import concurrent.futures
from utils.Dataset import utils as datasetUtils
import time
import pandas as pd

from_where_dict = {'Local': 0, 'Cloud': 0}

def create_required_directories_segmentation(dataset_root, dataset_limit):
    """
    Creates necessary directories for storing images and masks with a unique identifier.
    
    Args:
    dataset_root (str): The root directory name for the dataset.
    """
    unique_id = str(int(time.time()))
    base_path = f'{dataset_root}_{unique_id}'

    os.makedirs(os.path.dirname(f'data/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'data/csv/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'data/{base_path}/'), exist_ok=True)

    for mode in ['train', 'test']:
        os.makedirs(os.path.dirname(f'data/{base_path}/{mode}/'), exist_ok=True)
        os.makedirs(os.path.dirname(f'data/{base_path}/{mode}/images/'), exist_ok=True)
        os.makedirs(os.path.dirname(f'data/{base_path}/{mode}/mask/'), exist_ok=True)
    
    response = datasetUtils.read_data_from_postgres(dataset_root, column_type = 'polygon_labels' ,limit = dataset_limit)
    # print(response)
    
    if response is not None:
        pd.DataFrame(response).to_csv(f'data/csv/{base_path}.csv', index = False)
    else:
        raise ValueError("Failed to retrieve data from the database.")

    return base_path

def process_single_image_segmentation(image_data, dataset_length, dataset_root, print_interval, mode = 'train'):
    """
    Processes a single image: loads it, creates a mask, and saves both.
    
    Args:
    image_data (tuple): A tuple containing the index and row data of the image.
    dataset (DataFrame): The entire dataset DataFrame.
    dataset_root (str): The root directory name for the dataset.
    """
    index, row = image_data
    try:
        if index % print_interval == 0:
            print(f"{index} / {dataset_length}")

        serialise_dir = row['image_path'].replace('/', '_')
        mask_path = f"data/{dataset_root}/{mode}/mask/{serialise_dir}"
        image_name = f"data/{dataset_root}/{mode}/images/{serialise_dir}"
        
        if os.path.exists(image_name):
            return
        
        row['image_path'] = row['image_path'].replace('.png', '.jpeg')
        # print(row['image_path'])
        image, from_where = datasetUtils.get_image_from_source(row['image_path'])
        from_where_dict[from_where] += 1

        if not image:
            return
        pathology = '_'.join(dataset_root.split('_')[:-1])
        mask = datasetUtils.create_mask(image, datasetUtils.parse_polygonlabel(row['polygonlabels'], image.size, pathology))

        mask.save(mask_path)
        image.save(image_name)  

    except Exception as error:
        os.system(f"rm -rf data/{dataset_root}/{mode}/images/{serialise_dir}")
        os.system(f"rm -rf data/{dataset_root}/{mode}/mask/{serialise_dir}")
        print(f"An error occurred during image processing: {error}")


def execute_dataset_processing_segmentation(dataset_root, print_interval = 10, dataset_limit = 50):
    """
    Executes the processing of the entire dataset.
    
    Args:
    dataset_root (str): The root directory name for the dataset.
    """
    dataset_root = create_required_directories_segmentation(dataset_root, dataset_limit)
    dataset = datasetUtils.load_dataset_and_shuffle(dataset_root)

    train_dataset, test_dataset = datasetUtils.split_dataset(dataset, test_data_csv='')
    print(train_dataset.shape, test_dataset.shape)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image_segmentation, (index, row), len(train_dataset), dataset_root, print_interval) for index, row in train_dataset.iterrows()]
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image_segmentation, (index, row), len(test_dataset), dataset_root, print_interval, 'test') for index, row in test_dataset.iterrows()]
        concurrent.futures.wait(futures)

    total_images = from_where_dict['Local'] + from_where_dict['Cloud']
    local_percentage = (from_where_dict['Local'] / total_images) * 100 if total_images > 0 else 0
    global_percentage = (from_where_dict['Cloud'] / total_images) * 100 if total_images > 0 else 0
    print(f"{local_percentage:.2f}% are downloaded from Local and {global_percentage:.2f}% images from Cloud")

    return dataset_root

