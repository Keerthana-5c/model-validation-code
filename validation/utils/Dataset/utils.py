import ast, os, json
from PIL import Image, ImageDraw
import pandas as pd
import numpy as np
from utils.StorageOperations.postgres import execute_sql_query
from utils.StorageOperations import azure as azureUtils
from dotenv import load_dotenv

# Load environment variables from 'pwd.env'
load_dotenv('pwd.env')

def load_dataset_and_shuffle(dataset_root):
    """
    Loads the dataset from a CSV file and shuffles it.
    
    Args:
    dataset_root (str): The root directory name where the CSV file is located.
    
    Returns:
    DataFrame: A shuffled pandas DataFrame containing the dataset.
    """
    dataset = pd.read_csv(f'data/csv/{dataset_root}.csv')
    assert 'image_path' in dataset.columns and 'polygonlabels' in dataset.columns, "DataFrame must contain 'image_path' and 'polygonlabels' columns"
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    print(dataset)
    return dataset

def parse_polygonlabel(polygonlabel_annotation, image_size, target_label):
    """
    Parses polygon label formatted annotation to extract coordinates of polygons matching the target label.
    
    Args:
    polygonlabel_annotation (str): The polygon label formatted string.
    image_size (tuple): A tuple (width, height) of the image.
    target_label (str): The label of interest to extract coordinates for.
    
    Returns:
    list: A list of coordinates for the polygons that match the target label.
    """
    width, height = image_size
    try:
        polygon_data = json.loads(polygonlabel_annotation.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            polygon_data = ast.literal_eval(polygonlabel_annotation)
        except ValueError as e:
            print(f"Error parsing polygon label data: {e}")
            return []

    if not len(polygon_data): return [];

    coordinates = []
    for annotation in polygon_data:
        if annotation['polygonlabels'][0].lower() == target_label.replace('_', ' '):
            polygon_coords = []
            for point in annotation['points']:
                x = (point[0] / 100) * width
                y = (point[1] / 100) * height
                polygon_coords.append([x, y])
            coordinates.append(polygon_coords)

    return coordinates

def create_mask(image, polygons, save_path=None):
    """
    Creates a binary mask from polygon coordinates and applies it to the given image.
    
    Args:
    image (Image): The PIL Image object.
    polygons (list): A list of polygons, where each polygon is a list of (x, y) tuples.
    save_path (str, optional): Path to save the masked image. If None, the image is not saved.
    
    Returns:
    Image: A PIL Image object with the mask applied.
    """
    mask = Image.new('1', image.size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        # Ensure the polygon coordinates are in the correct format (tuples of integers)
        formatted_polygon = [(int(x), int(y)) for x, y in polygon]
        draw.polygon(formatted_polygon, outline=1, fill=1)
    del draw

    if save_path:
        mask.save(save_path)
    
    return mask

def rotate_point(original_x, original_y, center_x, center_y, angle_radians):
    """
    Rotates a point around a given center by a specified angle.

    Args:
    original_x (float): The x-coordinate of the original point.
    original_y (float): The y-coordinate of the original point.
    center_x (float): The x-coordinate of the rotation center.
    center_y (float): The y-coordinate of the rotation center.
    angle_radians (float): The rotation angle in radians.

    Returns:
    tuple: A tuple containing the new x and y coordinates after rotation and the differences in x and y.
    """
    x_shifted = original_x - center_x
    y_shifted = original_y - center_y
    
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
    y_rotated = x_shifted * sin_theta + y_shifted * cos_theta
    
    x_rotated += center_x
    y_rotated += center_y
    
    return int(abs(x_rotated)), int(abs(y_rotated)), int(original_x - x_rotated), int(original_y - y_rotated)

def rotate_bbox(top_left_x, top_left_y, width, height, angle_radians):
    """
    Rotates the bounding box defined by its top-left corner, width, and height.

    Args:
    top_left_x (float): The x-coordinate of the top-left corner of the bounding box.
    top_left_y (float): The y-coordinate of the top-left corner of the bounding box.
    width (float): The width of the bounding box.
    height (float): The height of the bounding box.
    angle_radians (float): The rotation angle in radians.

    Returns:
    list: A list of new coordinates for each corner of the rotated bounding box.
    """
    center_x = top_left_x + width / 2
    center_y = top_left_y + height / 2
    
    corners = [
        (top_left_x, top_left_y),
        (top_left_x + width, top_left_y),
        (top_left_x + width, top_left_y + height),
        (top_left_x, top_left_y + height)
    ]
    
    rotated_corners = []
    previous_x_shift = None
    previous_y_shift = None
    for corner_x, corner_y in corners:
        new_x, new_y, x_shift, y_shift = rotate_point(corner_x, corner_y, center_x, center_y, angle_radians)
        if previous_x_shift is None and previous_y_shift is None:
            previous_x_shift = x_shift
            previous_y_shift = y_shift
        rotated_corners.append([new_x + previous_x_shift, new_y + previous_y_shift])
    
    return rotated_corners



def read_data_from_postgres(pathology, column_type , project_name = None, limit = 'NULL', database_name = 'annotations'):
    """
    Reads data from a PostgreSQL database using a predefined SQL query stored in a file.

    Returns:
    DataFrame: A DataFrame containing the results of the SQL query.
    """
    sql_file_path = 'sql/get_data_for_chest.sql'

    with open(sql_file_path, 'r') as file:
        sql_query = file.read()

    condition = '1=1'
    # print(sql_query.format(pathology.replace('_', ' '), limit, column_type, condition))
    
    return execute_sql_query(sql_query.format(pathology.replace('_', ' '), limit, column_type, condition), database_name)


def get_image_from_source(image_path):
    local_image_path = os.path.join(os.getenv("LOCAL_DATA_DIR"), image_path.replace('/','_'))
    if os.path.exists(local_image_path):
        return Image.open(local_image_path), 'Local'
    else:
        return azureUtils.download_file('image', image_path, return_as_image=True), 'Cloud'

def split_dataset(dataset, train_ratio=0.9, test_data_csv=None):
    """
    Shuffles and splits the dataset into training and testing sets based on the specified ratio.

    Args:
    dataset (DataFrame): The dataset to be split.
    train_ratio (float): The proportion of the dataset to include in the train split.

    Returns:
    tuple: A tuple containing the training and testing DataFrames.
    """
    # Shuffle the dataset
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

    if test_data_csv:
       
        gt_test_data = pd.read_csv(test_data_csv)
        processed_series = (gt_test_data['image_path']).str.replace('png','jpeg').str.replace('_','/')
        
        test_dataset = shuffled_dataset[shuffled_dataset['image_path'].isin(processed_series)].reset_index(drop=True)
        test_need = int(len(shuffled_dataset)*0.1)-len(test_dataset)
        
        if test_need > 0:
          additional_test = shuffled_dataset[~shuffled_dataset['image_path'].isin(processed_series)].iloc[:test_need]
          test_dataset = test_dataset._append(additional_test).reset_index(drop=True)
        else:
          test_dataset = shuffled_dataset[shuffled_dataset['image_path'].isin(processed_series)].reset_index(drop=True)
    
        train_dataset = shuffled_dataset[~shuffled_dataset['image_path'].isin(test_dataset['image_path'])].reset_index(drop=True)

    else:
        train_size = int(len(shuffled_dataset)*train_ratio)
        test_dataset = shuffled_dataset.iloc[train_size:].reset_index(drop=True)
        train_dataset = shuffled_dataset.iloc[:train_size].reset_index(drop=True) 
        # print("shuffled:",len(shuffled_dataset),"train:",len(train_dataset),"test:",len(test_dataset))
        # print("shuffled_dataset:",shuffled_dataset)
        # print("train_dataset:",train_dataset)
        # print("test_dataset:",test_dataset)
        

    return train_dataset, test_dataset

