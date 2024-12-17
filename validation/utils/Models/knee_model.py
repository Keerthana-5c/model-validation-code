# import os
# import cv2
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from detectron2.config import get_cfg
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

# # Step 1: Load the Detectron2 Model
# def load_model(model_path, threshold=0.5):
#     cfg = get_cfg()
#     cfg.merge_from_file("/home/ai-user/test_models/knee_models/knee_models_loose_fab_tibia.yml")  # Replace with your model's config file path
#     cfg.MODEL.WEIGHTS = model_path
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
#     cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     return DefaultPredictor(cfg)

# # Step 2: Process Individual Image
# def process_image(predictor, image_path, image_name, output_folder):
#     img = cv2.imread(image_path)
#     outputs = predictor(img)

#     # Extract predictions
#     instances = outputs["instances"]
#     pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
#     pred_classes = instances.pred_classes.cpu().numpy()
#     scores = instances.scores.cpu().numpy()

#     # Store results
#     results = []
#     for box, cls, score in zip(pred_boxes, pred_classes, scores):
#         results.append({
#             "image": image_name,
#             "class": cls,
#             "score": score,
#             "bbox": box.tolist()
#         })

#     # Visualization
#     visualize_predictions(img, outputs, image_name, output_folder)
#     return results

# # Step 3: Visualize Predictions
# def visualize_predictions(img, outputs, image_name, output_folder):
#     v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("__unused"), scale=1.2)
#     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#     plt.figure(figsize=(15, 10))
#     plt.imshow(out.get_image()[:, :, ::-1])
#     plt.axis('off')

#     # Save matplotlib figure
#     output_path = os.path.join(output_folder, f"{image_name}_predictions.png")
#     plt.savefig(output_path)
#     plt.close()

# # Step 4: Validate Images with Multithreading
# def validate_images_multithread(predictor, test_folder, output_folder, output_csv, max_threads=16):
#     results = []
#     image_paths = [
#         (os.path.join(test_folder, image_name), image_name)
#         for image_name in os.listdir(test_folder)
#     ]

#     with ThreadPoolExecutor(max_threads) as executor:
#         futures = [
#             executor.submit(process_image, predictor, img_path, img_name, output_folder)
#             for img_path, img_name in image_paths
#         ]

#         for future in as_completed(futures):
#             results.extend(future.result())

#     # Save to CSV
#     df = pd.DataFrame(results)
#     df.to_csv(output_csv, index=False)

# # Step 5: Main Execution
# if __name__ == "__main__":
#     model_path = "/home/ai-user/test_models/knee_models/knee_models_loose_fab_tibia.pth"
#     test_folder = "/home/ai-user/test_models/cxr_images"
#     output_folder = "/home/ai-user/test_models/validation/utils/Models/ouput_folder"  # Folder to save visualizations
#     output_csv = "/home/ai-user/test_models/validation/utils/Models/predictions.csv"

#     os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

#     predictor = load_model(model_path)
#     validate_images_multithread(predictor, test_folder, output_folder, output_csv, max_threads=16)


import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Step 1: Load the Detectron2 Model
def load_model(model_path, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file("/home/ai-user/test_models/knee_models/knee_models_loose_fab_tibia.yml")  # Replace with your model's config file path
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Step 2: Validate Images
def validate_images(predictor, test_folder, output_csv):
    results = []
    for image_name in os.listdir(test_folder):
        image_path = os.path.join(test_folder, image_name)
        img = cv2.imread(image_path)
        outputs = predictor(img)

        # Extract predictions
        instances = outputs["instances"]
        pred_boxes = instances.pred_boxes.tensor.cpu().numpy()
        pred_classes = instances.pred_classes.cpu().numpy()
        scores = instances.scores.cpu().numpy()

        # Save predictions in CSV
        for box, cls, score in zip(pred_boxes, pred_classes, scores):
            results.append({
                "image": image_name,
                "class": cls,
                "score": score,
                "bbox": box.tolist()
            })

        # Visualization
        visualize_predictions(img, outputs, image_name, "output_folder")  # Specify an output folder

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

# Step 3: Visualize Predictions
def visualize_predictions(img, outputs, image_name, output_folder):
    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("__unused"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(15, 10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')

    # Save matplotlib figure
    output_path = os.path.join(output_folder, f"{image_name}_predictions.png")
    plt.savefig(output_path)
    plt.close()

# Step 4: Main Execution
if __name__ == "__main__":
    model_path = "path/to/model.pth"
    test_folder = "path/to/test/images"
    output_csv = "predictions.csv"

    predictor = load_model(model_path)
    validate_images(predictor, test_folder, output_csv)