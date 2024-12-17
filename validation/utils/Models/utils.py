import torch, time
import csv
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image, ImageDraw
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import precision_score, recall_score, accuracy_score, jaccard_score, f1_score
import os
import pandas as pd
from ultralytics import YOLO


def gradcam(model, target_layers, image, input, flag):
    input_tensor = input.unsqueeze(0)
    if flag == 'segmentation':
        output = model(input_tensor)
        mask = output[0].cpu().detach().numpy()
        
        class SemanticSegmentationTarget:
            def __init__(self, mask):
                self.mask = torch.from_numpy(mask)
                if torch.cuda.is_available():
                    self.mask = self.mask.cuda()
            def __call__(self, model_output):
                return (model_output[ 0, :, :]  * self.mask).sum()
        
        targets_lay = [SemanticSegmentationTarget(mask)]
    else:
        targets_lay = [ClassifierOutputTarget(1)]

    rgb_img = np.array(image) / 255

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets_lay)[0, :]
        cam_image = show_cam_on_image(rgb_img.transpose(1, 2, 0), grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_image)


def train_model(model, train_loader, test_loader, loss_fn, eval_fn, optimizer, epochs, output_dir, device, scheduler, status=True):
    train_losses, test_losses = [], []
    train_scores, test_scores = [], []
    epoch_indices = []
    
    log_writer = SummaryWriter(f'data/{output_dir}/logs/tensorboard')
    min_test_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_score = 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            score = eval_fn(torch.sigmoid(outputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_score += score.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_score = train_score / len(train_loader)

        model.eval()
        test_loss, test_score = 0, 0
        with torch.no_grad():
            for inputs, targets, _ in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                test_loss += loss.item()
                score = eval_fn(torch.sigmoid(outputs), targets)
                test_score += score.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_score = test_score / len(test_loader)

        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            torch.save(model.state_dict(), f'data/{output_dir}/checkpoint/best_model_{epoch}_{avg_test_score:.4f}.pt')

        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        if status:
            print(f"Epoch {epoch}/{epochs} => Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f} - Train Score: {avg_train_score:.4f} - Test Score: {avg_test_score:.4f} - Time: {epoch_time:.2f} sec")

        log_writer.add_scalars('Loss', {'train': avg_train_loss, 'test': avg_test_loss}, epoch)
        log_writer.add_scalars('Score', {'train': avg_train_score, 'test': avg_test_score}, epoch)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_scores.append(avg_train_score)
        test_scores.append(avg_test_score)
        epoch_indices.append(epoch)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_indices, train_losses, label='Train Loss')
    plt.plot(epoch_indices, test_losses, label='Test Loss')
    plt.xlabel('Epoch'), plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch_indices, train_scores, label='Train Score')
    plt.plot(epoch_indices, test_scores, label='Test Score')
    plt.xlabel('Epoch'), plt.ylabel('Score')
    plt.legend()

    plt.savefig(f'data/{output_dir}/logs/loss.png')
    torch.save(model, f'data/{output_dir}/final_model.pt')
    log_writer.close()

    return model

def calculate_metrics(predicted_mask, ground_truth):

    predicted_mask = np.asarray(predicted_mask).flatten()
    ground_truth = np.asarray(ground_truth).flatten()
    

    predicted_mask = predicted_mask.astype(int)
    ground_truth = ground_truth.astype(int)
    
    iou = jaccard_score(ground_truth, predicted_mask, average='macro')
    dice_score = f1_score(ground_truth, predicted_mask, average='macro')
    precision = precision_score(ground_truth, predicted_mask, average='macro', zero_division=0)
    recall = recall_score(ground_truth, predicted_mask, average='macro', zero_division=0)
    accuracy = accuracy_score(ground_truth, predicted_mask)
    
    return iou, dice_score, precision, recall, accuracy

def validate_model(model, val_loader, loss_fn, eval_fn, output_dir, device, target_layers, flag):
    model.eval()
    total_loss, total_score = 0, 0
    all_iou, all_dice, all_precision, all_recall, all_accuracy = [], [], [], [], []
    results = []
    result_1 = []
    output_path = f'data/{output_dir}/results/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    csv_path = f'data/{output_dir}/'
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, image_path) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            score = eval_fn(torch.sigmoid(outputs), targets)
            total_score += score.item()

            for img_idx in range(targets.size(0)):
                predicted_mask = (torch.sigmoid(outputs[img_idx]) > 0.9).cpu().numpy().astype(int)
                ground_truth = targets[img_idx].cpu().numpy().astype(int)
                iou, dice_score, precision, recall, accuracy = calculate_metrics(predicted_mask, ground_truth)
                
                all_iou.append(iou)
                all_dice.append(dice_score)
                all_precision.append(precision)
                all_recall.append(recall)
                all_accuracy.append(accuracy)
                
                image_name = image_path[img_idx]
                ground_truth_value = 1 if np.any(ground_truth) else 0
                prediction_value = 1 if np.any(predicted_mask) else 0
                result_1.append([image_name,ground_truth_value,prediction_value])
                
                results.append([image_path[img_idx], ground_truth.flatten(), predicted_mask.flatten(), iou, dice_score, precision, recall, accuracy])
                
                if flag == 'segmentation':
                    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

                predicted_mask = torch.sigmoid(outputs[img_idx]) > 0.9
                original_img = inputs[img_idx].cpu().squeeze().numpy()
                normalized_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
                plt_index = 0
                ax[plt_index].imshow(np.transpose(normalized_img, (1, 2, 0)), cmap='gray')
                ax[plt_index].set_title('Original')
                ax[plt_index].axis('off')
                plt_index += 1

                if flag == 'segmentation':
                    ax[plt_index].imshow(targets[img_idx].cpu().squeeze(), cmap='gray')
                    ax[plt_index].set_title('Target')
                    ax[plt_index].axis('off')
                    plt_index += 1

                    ax[plt_index].imshow(predicted_mask.cpu().squeeze(), cmap='gray')
                    ax[plt_index].set_title('Predicted')
                    ax[plt_index].axis('off')
                    plt_index += 1

                with torch.enable_grad():
                    ax[plt_index].imshow(gradcam(model, target_layers, normalized_img, inputs[img_idx], flag), cmap='gray')
                    ax[plt_index].set_title('Grad CAM')
                    ax[plt_index].axis('off')
                
                plt.savefig(f'{output_path}{image_path[img_idx]}.png')
                plt.close()
                
    csv_file_path = os.path.join(csv_path, 'validation_results.csv')
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Image Name', 
            'Ground Truth', 
            'Prediction'
        ])
        writer.writerows(result_1)        
    avg_loss = total_loss / len(val_loader)
    avg_iou = np.mean(all_iou)
    avg_dice = np.mean(all_dice)
    avg_precision = np.mean(all_precision)
    avg_recall = np.mean(all_recall)
    avg_accuracy = np.mean(all_accuracy)

    # print(f'Validation Loss: {avg_loss:.4f}, Validation Score: {avg_score:.4f}')
    print(f'Average IoU: {avg_iou:.4f}')
    print(f'Average Dice Score: {avg_dice:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    pass

def bbox_prediction(dataset_root):
    model_path = f'data/{dataset_root}/logs/train/weights/best.pt'
    test_folder = f'data/{dataset_root}/test/images'
    output_folder = f'data/{dataset_root}/prediction/images'
    csv_file_path = f'data/{dataset_root}/prediction/report.csv'
    label_folder = f'data/{dataset_root}/test/labels'

    model = YOLO(model_path)

    os.makedirs(output_folder, exist_ok=True)

    csv_data = []

    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)

        original_img = Image.open(img_path).convert('RGB')
        img_width, img_height = original_img.size 

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_folder, label_name)
        
        target_img_pil = original_img.copy()

        ground_truth_exists = False

        if os.path.exists(label_path):
            with open(label_path, 'r') as label_file:
                lines = label_file.readlines()
                if len(lines) > 0:
                    for line in lines:
                        data = list(map(float, line.strip().split()))
                        if len(data) >= 8: 
                            ground_truth_exists = True
                            class_index = int(data[0])
                            x1, y1 = data[1] * img_width, data[2] * img_height
                            x2, y2 = data[3] * img_width, data[4] * img_height
                            x3, y3 = data[5] * img_width, data[6] * img_height
                            x4, y4 = data[7] * img_width, data[8] * img_height

                            draw = ImageDraw.Draw(target_img_pil)
                            draw.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], outline="green", width=13)
        else:
            print(f"Label file not found or empty for {img_name}")

        results = model(img_path)
        

        prediction_status = "True" if results[0].obb.xyxy is not None and len(results[0].obb.xyxy) >0 else "False"
        print(prediction_status,ground_truth_exists)

        csv_data.append([img_name,ground_truth_exists,prediction_status])

        predicted_img_np = results[0].plot()

        original_img_np = np.array(original_img)
        target_img_np = np.array(target_img_pil)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_img_np)
        axes[0].set_title("Original", fontsize=21)
        axes[0].axis('off')

        axes[1].imshow(target_img_np)
        axes[1].set_title("Target", fontsize=21)
        axes[1].axis('off')

        axes[2].imshow(predicted_img_np)
        axes[2].set_title("Predicted", fontsize=21)
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, img_name))
        plt.close()

        df = pd.DataFrame(csv_data,columns=['image_path','ground_truth','prediction'])
        df.to_csv(csv_file_path, index=False)