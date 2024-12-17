from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv

load_dotenv('pwd.env')

def upload_file(cloud, file_path, bucket_name, object_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        client.fput_object(bucket_name, object_name, file_path)

        print("File uploaded successfully.")

    except S3Error as err:
        print(f"An error occurred while uploading the file: {err}")

def download_file(cloud, bucket_name, object_name, download_path):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        client.fget_object(bucket_name, object_name, download_path)

        print("File downloaded successfully.")

    except S3Error as err:
        print(f"An error occurred while downloading the file: {err}")

def delete_file(cloud, bucket_name, object_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        client.remove_object(bucket_name, object_name)

        print("File deleted successfully.")

    except S3Error as err:
        print(f"An error occurred while deleting the file: {err}")

def create_bucket(cloud, bucket_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        print(f"Bucket '{bucket_name}' created successfully.")

    except S3Error as err:
        print(f"An error occurred while creating the bucket: {err}")

def delete_bucket(cloud, bucket_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        client.remove_bucket(bucket_name)

        print(f"Bucket '{bucket_name}' deleted successfully.")

    except S3Error as err:
        print(f"An error occurred while deleting the bucket: {err}")

def list_objects(cloud, bucket_name, directory_path):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        objects = client.list_objects(bucket_name, prefix=directory_path, recursive=True)
        object_names = [obj.object_name for obj in objects]

        return object_names

    except S3Error as err:
        print(f"An error occurred while listing objects: {err}")

def copy_object(cloud, source_bucket_name, source_object_name, destination_bucket_name, destination_object_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        # Copy object
        client.copy_object(destination_bucket_name, destination_object_name, f"{source_bucket_name}/{source_object_name}")

        print(f"Object '{source_object_name}' copied from '{source_bucket_name}' to '{destination_bucket_name}/{destination_object_name}' successfully.")

    except S3Error as err:
        print(f"An error occurred while copying the object: {err}")

def move_object(cloud, source_bucket_name, source_object_name, destination_bucket_name, destination_object_name):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        # Copy object
        client.copy_object(destination_bucket_name, destination_object_name, f"{source_bucket_name}/{source_object_name}")

        # Remove original object
        client.remove_object(source_bucket_name, source_object_name)

        print(f"Object '{source_object_name}' moved from '{source_bucket_name}' to '{destination_bucket_name}/{destination_object_name}' successfully.")

    except S3Error as err:
        print(f"An error occurred while moving the object: {err}")

def list_buckets(cloud):
    try:
        endpoint = os.getenv(f"{cloud}_ENDPOINT")
        access_key = os.getenv(f"{cloud}_ACCESS_KEY")
        secret_key = os.getenv(f"{cloud}_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        buckets = client.list_buckets()
        bucket_names = [bucket.name for bucket in buckets]

        return bucket_names

    except S3Error as err:
        print(f"An error occurred while listing buckets: {err}")

def directory_exists(bucket_name, directory_path):
    try:
        endpoint = os.getenv("MINIO_ENDPOINT")
        access_key = os.getenv("MINIO_ACCESS_KEY")
        secret_key = os.getenv("MINIO_SECRET_KEY")

        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=True)

        objects = client.list_objects(bucket_name, prefix=directory_path, recursive=True)

        for obj in objects:
            if obj.object_name.startswith(directory_path):
                return True
        
        return False
    
    except S3Error as err:
        print(f"An error occurred while checking if directory exists: {err}")
        return False

