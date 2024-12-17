import concurrent.futures
from utils.Dataset import utils as datasetUtils
import time, math, os, ast
import pandas as pd

from_where_dict = {'Local': 0, 'Cloud': 0}


def map_classes_to_indices(class_list):
    class_to_index = {class_name: index for index, class_name in enumerate(class_list)}
    index_to_classname = {index : str(class_name) for index, class_name in enumerate(class_list)}
    return class_to_index, index_to_classname

def setup_dataset_directories(dataset_root, dataset_limit):
    """
    Set up directories for storing dataset images and labels, each with a unique timestamp.
    
    Args:
    dataset_root (str): Base name for the dataset directory.
    """
    timestamp = str(int(time.time()))
    base_directory = f'{dataset_root}_{timestamp}'

    os.makedirs('data/', exist_ok=True)
    os.makedirs('data/csv/', exist_ok=True)

    os.makedirs(f'data/{base_directory}/train/images', exist_ok=True)
    os.makedirs(f'data/{base_directory}/train/labels', exist_ok=True)
    os.makedirs(f'data/{base_directory}/test/images', exist_ok=True)
    os.makedirs(f'data/{base_directory}/test/labels', exist_ok=True)

    data_response = datasetUtils.read_data_from_postgres(dataset_root, column_type='bbox', limit=dataset_limit)
    if data_response is not None:
        pd.DataFrame(data_response).to_csv(f'data/csv/{base_directory}.csv', index=False)
    else:
        raise ValueError("Failed to retrieve data from the database.")

    return base_directory


def collect_and_write_image_paths(root_path, mode='train'):
    """
    Collects image paths from a specified directory and writes them to a text file if corresponding label files exist.

    Args:
    root_path (str): The root directory path where the images and labels are stored.
    mode (str): The subdirectory within the root path that specifies the dataset type (e.g., 'train' or 'test').
    """
    image_files = []
    image_dir = f'data/{root_path}/{mode}/images/'
    label_dir = f'data/{root_path}/{mode}/labels/'
    output_file = f'data/{root_path}/{mode}/{mode}.txt'

    # Collect all image files that have a corresponding label file
    for filename in os.listdir(image_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            label_file_path = f'{label_dir}{filename.split(".")[0]}.txt'
            # print(label_file_path)
            if os.path.exists(label_file_path):
                image_files.append(f'{image_dir}{filename}')

    # Write the collected image paths to the output file
    with open(output_file, "w") as outfile:
        for image in image_files:
            outfile.write(f"{image}\n")

    # Output statistics about the files
    total_images = len(os.listdir(image_dir))
    total_labels = len(os.listdir(label_dir))
    with open(output_file, 'r') as file:
        total_lines = len(file.readlines())

    print(f"Total images in {mode} directory: {total_images}")
    print(f"Total labels in {mode} directory: {total_labels}")
    print(f"Total lines written in {mode}.txt: {total_lines}")


def process_image_boundingbox(image_data, dataset_length, dataset_root, print_interval, class_indices, index_to_classname, mode='train'):
    """
    Process and save bounding box data for a single image.
    
    Args:
    image_data (tuple): Contains the index and row data of the image.
    dataset_length (int): Total length of the dataset.
    dataset_root (str): The root directory for the dataset.
    print_interval (int): Interval at which to print progress.
    class_indices (dict): Dictionary mapping label names to class indices.
    index_to_classname (dict): Counter dictionary for tracking label occurrences.
    mode (str): Mode of the dataset, either 'train', 'test', etc.
    """
    index, row = image_data

    if index % print_interval == 0:
        print(f"{mode} => {index} / {dataset_length}")
    
    saved_images_count = 0
    row['image_path'] = row['image_path'].replace('.png', '.jpeg')
    image_file_path = f'data/{dataset_root}/{mode}/images/{row["image_path"].replace("/","_")}'
    label_file_path = f'data/{dataset_root}/{mode}/labels/{row["image_path"].replace("/","_").replace(".jpeg", ".txt")}'

    if not os.path.exists(image_file_path) and not os.path.exists(label_file_path):
        image, from_where = datasetUtils.get_image_from_source(row['image_path'])
        from_where_dict[from_where] += 1        
        
        if not image:
            print("Image not found")
            return
        
        image.save(image_file_path)
        saved_images_count += 1

        if not row['polygonlabels'] or row['polygonlabels'] == '[]':
            with open(label_file_path, 'w') as label_file:
                pass

        else:
            bounding_data = ast.literal_eval(row['polygonlabels'])
            # print(f"{row['image_path']} has {len(bounding_data)}")
            
            if bounding_data:
                with open(label_file_path, 'a') as label_file:
                    image_width, image_height = image.size

                    for item in bounding_data:
                        label = item['rectanglelabels'][0].lower()

                        class_index = class_indices.get(label, class_indices['non mediastinal mass'])

                        if label in class_indices:
                            x = item['x'] / 100
                            y = item['y'] / 100
                            height = item['height'] / 100
                            width = item['width'] / 100

                            angle_radians = math.radians(item['rotation'])
                            rotated_corners = datasetUtils.rotate_bbox(
                                int(x * image_width), int(y * image_height),
                                int(width * image_width), int(height * image_height),
                                angle_radians
                            )

                            normalized_corners = [(x / image_width, y / image_height) for x, y in rotated_corners]
                            flattened_corners = [coord for corner in normalized_corners for coord in corner]

                            # print(f"Bounding box corners: {rotated_corners} for {row['image_path']}")
                            # print(f"Normalized corners: {normalized_corners} for {row['image_path']}")
                            # print(f"Flattened corners: {flattened_corners} for {row['image_path']}")

                            try:
                                if flattened_corners:
                                    label_content = f'{class_index} {" ".join(map(str, flattened_corners))}\n'
                                    with open(label_file_path,'a') as file:
                                        file.write(label_content)
                                    # print(f"Appending bounding box {flattened_corners} for label {label} to {label_file_path}")
                                    # index_to_classname[label] +=    1  
                                else:
                                    print("No valid bounding box data to write.")
                            except Exception as e:
                                print(f"error occuring : {e}")

    else:
        print('File already exists')


def execute_boundingbox_processing(dataset_root, class_list, print_interval=100, dataset_limit='NULL'):
    """
    Orchestrates the processing of bounding box data for the entire dataset.
    
    Args:
    dataset_root (str): The root directory name for the dataset.
    """
    dataset_root = setup_dataset_directories(dataset_root, dataset_limit)
    dataset = datasetUtils.load_dataset_and_shuffle(dataset_root)
    train_dataset, test_dataset = datasetUtils.split_dataset(dataset, train_ratio=0.8)

    class_indices, index_to_classname = map_classes_to_indices(class_list)
    print(class_indices, index_to_classname)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_boundingbox, (index, row), len(train_dataset), dataset_root, print_interval, class_indices, index_to_classname) for index, row in train_dataset.iterrows()]
        concurrent.futures.wait(futures)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_boundingbox, (index, row), len(test_dataset), dataset_root, print_interval, class_indices, index_to_classname, mode='test') for index, row in test_dataset.iterrows()]
        concurrent.futures.wait(futures)

    collect_and_write_image_paths(dataset_root)
    collect_and_write_image_paths(dataset_root, 'test')

    total_images = from_where_dict['Local'] + from_where_dict['Cloud']
    local_percentage = (from_where_dict['Local'] / total_images) * 100 if total_images > 0 else 0
    global_percentage = (from_where_dict['Cloud'] / total_images) * 100 if total_images > 0 else 0
    print(f"{local_percentage:.2f}% are downloaded from Local and {global_percentage:.2f}% images from Cloud")
    

    return dataset_root, index_to_classname

