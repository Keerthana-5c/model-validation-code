import os
import pandas as pd
import numpy as np
from minio import Minio
from minio.error import S3Error
from pydicom import dcmread
from PIL import Image
import csv
import concurrent.futures as futures
from dotenv import load_dotenv
import pylibjpeg
import pydicom as pydicom

load_dotenv('pwd.env')

bucket_name = "5cnetwork-newserver-dicom"
local_image_dir = "/home/ai-user/test_models/cxr_images/"

os.makedirs(local_image_dir, exist_ok=True)

def directory_exists(cloud, bucket_name, directory_path):
    try:
        endpoint = os.getenv(f'{cloud}_ENDPOINT')
        access_key = os.getenv(f'{cloud}_ACCESS_KEY')
        secret_key = os.getenv(f'{cloud}_SECRET_KEY')

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)
        objects = client.list_objects(bucket_name, prefix=directory_path, recursive=True)

        for obj in objects:
            if obj.object_name.startswith(directory_path):
                return True
        
        return False
    
    except S3Error as err:
        print(f"An error occurred while checking if directory exists in {cloud}: {err}")
        return False
    
def list_objects(cloud, bucket_name, directory_path):
    try:
        endpoint = os.getenv(f'{cloud}_ENDPOINT')
        access_key = os.getenv(f'{cloud}_ACCESS_KEY')
        secret_key = os.getenv(f'{cloud}_SECRET_KEY')

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)
        objects = client.list_objects(bucket_name, prefix=directory_path, recursive=True)
        return objects, client

    except S3Error as err:
        print(f"An error occurred while listing objects in {cloud}: {err}")
    
def check_source(path):
    try:
        if directory_exists("YOTTA", bucket_name, path):
            print("Images are in Yotta bucket...")
            lists, session = list_objects("YOTTA", bucket_name, path)
            return lists, session, "YOTTA"
    
        elif directory_exists("E2E", bucket_name, path):
            print("Images are in E2E bucket...")
            lists, session = list_objects("E2E", bucket_name, path)
            return lists, session, "E2E"
    
        else:
            print("Images are not present in both Yotta and E2E")
            return [], "Nothing", "Nothing"

    except Exception as e:
        print(f"An error occurred while checking the source: {e}")
        return [], "Nothing", "Nothing"

def convert_dicom_to_image(dcm):
    pixel_array = dcm.pixel_array.astype(float)
    if 'PhotometricInterpretation' in dcm and dcm.PhotometricInterpretation == 'MONOCHROME1':
        pixel_array = np.max(pixel_array) - pixel_array
    scaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
    return Image.fromarray(np.uint8(scaled_image))

def migrate(path):
    try:
        actual_path = path + "/"

        objects, session, cloud = check_source(actual_path)
        
        if cloud != "Nothing":
            for obj in objects:
                print(obj.object_name)
                obj = obj.object_name
    
                # Get the DICOM data from Yotta/E2E bucket
                data = session.get_object('5cnetwork-newserver-dicom', obj)
                dicom_bytes = b""
                for d in data.stream(64 * 1024):
                    dicom_bytes += d
            
                ds = pydicom.dcmread(pydicom.filebase.DicomBytesIO(dicom_bytes))
                print(ds.Modality)
                
                if ds.Modality in ('DX', 'CR'):
                    # Convert DICOM to image
                    image = convert_dicom_to_image(ds)
                    if image:
                        # Save image locally
                        formatted_name = obj.replace('/', '_') + '.jpeg'
                        image_path = os.path.join(local_image_dir, formatted_name)
                        image.save(image_path)
                        print(f"{image_path} is saved successfully locally.")
                
                else:
                    print("No valid modality found in DICOM file.")
    
    except Exception as e:
        print(f"An error occurred during migration: {e}")

def image_migration(input_csv):
    studies = pd.read_csv(input_csv)
    studies = studies[(studies["path"] != "") & (studies["path"].notna())]
    studies['status'] = None
    studies['error'] = None
    csv_name = 'csv/download_status.csv'

    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(studies.columns.tolist())

    def multi_thread(i, row):
        try:
            path = row['path']
            migrate(path)
            studies.at[i, "status"] = 'migrated'
        except Exception as e:
            studies.at[i, "status"] = 'failed'
            studies.at[i, "error"] = str(e)
        finally:
            with open(csv_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(studies.iloc[i].tolist())

    with futures.ThreadPoolExecutor(max_workers=32) as executors:
        to_do = []
        for i, row in studies.iterrows():
            future = executors.submit(multi_thread, i, row)
            to_do.append(future)

    for future in futures.as_completed(to_do):
        future.result()

def count_images_in_local_directory():
    """Count and print the number of images in the local image directory."""
    num_images = len([f for f in os.listdir(local_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    print(f"Number of images in '{local_image_dir}': {num_images}")

if __name__ == "__main__":
    count_images_in_local_directory()