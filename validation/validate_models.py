import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO  
from utils.Models import utils as modelUtils

class NormalizeAndReplaceNaN:
    def __init__(self, mean=0.5, std=0.5):  
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        
        tensor = (tensor - self.mean) / (self.std + 1e-8)
        
        
        tensor[torch.isnan(tensor)] = 0
        return tensor

normalize = NormalizeAndReplaceNaN(mean=0.5, std=0.5) 

# Define a unified transformation 
def get_transform(target_size=(224, 224)):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        normalize
    ])

# Function to validate segmentation model
def validate_segmentation_model(model, image_folder, output_dir, target_size=(224, 224)):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    results = []
    transform = get_transform(target_size)

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations to the image
        image_tensor = transform(image).unsqueeze(0).to(device)

        image = image.resize(target_size, Image.Resampling.LANCZOS)


        torch.cuda.empty_cache()

        with torch.no_grad():
            output = model(image_tensor)
            predicted_mask = (torch.sigmoid(output) > 0.5).cpu().numpy()

        # Ensure 2D mask extraction
        if predicted_mask.ndim == 4:  # Batch size and channel dimensions
            predicted_mask = predicted_mask[0, 0]  
        elif predicted_mask.ndim == 3:  # Single channel dimension
            predicted_mask = predicted_mask[0]

        # Visualization and saving
        pred_image = Image.fromarray((predicted_mask * 255).astype(np.uint8), mode='L')
        plt.imshow(image,cmap='gray')
        plt.imshow(pred_image,cmap='jet',alpha=0.5)
        plt.axis('off')

        fig = plt.gcf()  # Get the current figure
        fig.canvas.draw()  # Render the figure

        # Convert the canvas to a NumPy array
        plot_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plot_array = plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pred_image = Image.fromarray(plot_array)
        pred_image = pred_image.resize(target_size, Image.Resampling.LANCZOS)
        combined_image = Image.new('RGB', (image.width * 2, image.height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(pred_image.convert('RGB'), (image.width, 0))

        # Save the combined image
        combined_image.save(os.path.join(output_dir, f"{img_name}"))

        # Store prediction results
        prediction_value = 1 if np.any(predicted_mask) else 0
        results.append([img_name, prediction_value])


    # Save the results to a CSV file
    results_df = pd.DataFrame(results, columns=["Image Name", "Prediction"])
    results_df.to_csv('csv/prediction_results.csv', index=False)

    print("Segmentation validation complete. Results saved.")



# Function to validate YOLO model
def validate_yolo_model(model, image_folder, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    results = []  

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        image = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            predictions = model(image)  # Get predictions

        # Check if there are any predictions
        has_prediction = len(predictions[0]) > 0 if predictions else False

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        if has_prediction:
            predicted_img_np = predictions[0].plot()
            axes[1].imshow(predicted_img_np)
            axes[1].set_title("Predicted Image")
        else:
            axes[1].imshow(image)
            axes[1].set_title("No Prediction")
        
        axes[1].axis('off')

        
        comparison_img_path = os.path.join(output_dir, f'{img_name}')
        plt.savefig(comparison_img_path)
        plt.close()  

       
        results.append([img_name, 1 if has_prediction else 0])

   
    results_df = pd.DataFrame(results, columns=["Image Name", "Prediction"])
    results_df.to_csv('csv/prediction_results.csv', index=False)

# Main function to call validation based on model type
def main(model_path, image_folder, model_type, mapping_column):
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([set])
   
    try:
        if model_type == 'segmentation':
            # Load the segmentation model with PyTorch
            if os.path.isfile(model_path):
                model = torch.load(model_path, map_location=device)
                model.to(device)
                model.eval()
                validate_segmentation_model(model, image_folder, f'results/segmentation/{mapping_column}')
            else:
                print(f"Model weights not found at {model_path}. Please check the path.")
                return

        elif model_type == 'yolo':
            # Load the YOLO model directly using YOLO's library
            if os.path.isfile(model_path):
                model = YOLO(model_path,task='predict').to(device)
                validate_yolo_model(model, image_folder, f'results/yolo/{mapping_column}')
            else:
                print(f"Model weights not found at {model_path}. Please check the path.")
                return

        else:
            raise ValueError("Model type must be 'segmentation' or 'yolo'.")
   
    except Exception as e:
        print(f"Error loading model: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate models.')
    parser.add_argument('model_path', type=str, help='Path to the model weights')
    parser.add_argument('image_folder', type=str, help='Path to the folder containing images for validation')
    parser.add_argument('model_type', type=str, choices=['segmentation', 'yolo'], help='Type of model to validate')
    parser.add_argument('mapping_column', type=str, help='pathology name')

    args = parser.parse_args()
    main(args.model_path, args.image_folder, args.model_type, args.mapping_column)
