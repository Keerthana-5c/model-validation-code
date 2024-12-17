from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
from PIL import Image
import io

# Load environment variables from 'pwd.env'
load_dotenv('pwd.env')

def upload_file(file_path, container_name, blob_name):
    """Uploads a file to an Azure Blob Storage container."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Open the file and upload it to the blob
        with open(file_path, "rb") as file_data:
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(file_data)
            
        print("File uploaded successfully.")
        
    except Exception as error:
        print(f"An error occurred while uploading the file: {error}")

def download_file(container_name, blob_name, download_path = None, return_as_image=False):
    """Downloads a file from an Azure Blob Storage container."""
    # Ensure the directory exists
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        # Get the blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Download the blob data
        blob_data = blob_client.download_blob().readall()
        
        # If return_as_image is True, return an image, otherwise write to a file
        if return_as_image:
            image = Image.open(io.BytesIO(blob_data))
            return image
        else:
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            with open(download_path, "wb") as file:
                file.write(blob_data)
            
            print("File downloaded successfully.")
    
    except Exception as error:
        print(f"An error occurred while downloading the file: {error} - {blob_name}")
        return None

def delete_file(container_name, blob_name):
    """Deletes a file from an Azure Blob Storage container."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)
        # Get the blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Delete the blob
        blob_client.delete_blob()

        print("File deleted successfully.")
    
    except Exception as error:
        print(f"An error occurred while deleting the file: {error}")

def create_container(container_name):
    """Creates a new container in Azure Blob Storage."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create the container
        container_client = blob_service_client.create_container(container_name)

        print(f"Container '{container_name}' created successfully.")
    
    except Exception as error:
        print(f"An error occurred while creating the container: {error}")

def delete_container(container_name):
    """Deletes a container from Azure Blob Storage."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)

        # Delete the container
        container_client.delete_container()

        print(f"Container '{container_name}' deleted successfully.")
    
    except Exception as error:
        print(f"An error occurred while deleting the container: {error}")

def list_blobs(container_name, directory_path):
    """Lists all blobs in a specified directory within a container."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)

        # List all blobs that start with the directory path
        blob_list = container_client.list_blobs(name_starts_with=directory_path)

        # Extract blob names from the list
        blob_names = [blob.name for blob in blob_list]

        return blob_names
    
    except Exception as error:
        print(f"An error occurred while listing blobs: {error}")
        return []

def copy_blob(source_container_name, source_blob_name, destination_container_name, destination_blob_name):
    """Copies a blob from one container to another."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the source and destination blob clients
        source_blob_client = blob_service_client.get_blob_client(
            container=source_container_name, blob=source_blob_name)
        destination_blob_client = blob_service_client.get_blob_client(
            container=destination_container_name, blob=destination_blob_name)

        # Start copying the blob from the source URL
        destination_blob_client.start_copy_from_url(source_blob_client.url)

        print(f"Blob '{source_blob_name}' copied from '{source_container_name}' to '{destination_container_name}/{destination_blob_name}' successfully.")
    
    except Exception as error:
        print(f"An error occurred while copying the blob: {error}")

def move_blob(source_container_name, source_blob_name, destination_container_name, destination_blob_name):
    """Moves a blob from one container to another by copying and then deleting the original."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the source and destination blob clients
        source_blob_client = blob_service_client.get_blob_client(
            container=source_container_name, blob=source_blob_name)
        destination_blob_client = blob_service_client.get_blob_client(
            container=destination_container_name, blob=destination_blob_name)

        # Start copying the blob from the source URL
        destination_blob_client.start_copy_from_url(source_blob_client.url)
        # Delete the original blob
        source_blob_client.delete_blob()

        print(f"Blob '{source_blob_name}' moved from '{source_container_name}' to '{destination_container_name}/{destination_blob_name}' successfully.")
    
    except Exception as error:
        print(f"An error occurred while moving the blob: {error}")

def list_containers():
    """Lists all containers in the Azure Blob Storage."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # List all containers
        containers = blob_service_client.list_containers()
        container_names = [container.name for container in containers]

        return container_names
    
    except Exception as error:
        print(f"An error occurred while listing containers: {error}")
        return []

def directory_exists(container_name, directory_path):
    """Checks if a directory exists within a container."""
    try:
        # Retrieve connection string from environment variables
        connection_string = os.getenv("AZURE_CONN_STR")
        # Create a BlobServiceClient using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name)

        # List all blobs that start with the directory path
        blobs = container_client.list_blobs(name_starts_with=directory_path)
        for blob in blobs:
            if blob.name.startswith(directory_path):
                return True

        return False
    
    except Exception as error:
        print(f"An error occurred while checking if directory exists: {error}")
        return False