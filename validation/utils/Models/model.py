import os
from torch.utils.data import DataLoader
from utils.Models import utils as modelUtils
from ultralytics import YOLO
import yaml
import os


def run(root, flag,  model, dataloader, transformations, 
                     target_layers, loss_function, 
                     eval_function, optimizer, device, 
                     scheduler = None, batch_size= 4, num_epochs=10):
    os.makedirs(os.path.dirname(f'data/{root}/checkpoint/'), exist_ok=True)
    os.makedirs(os.path.dirname(f'data/{root}/results/'), exist_ok=True)

    if flag == 'segmentation':
        training_dataset = dataloader(f'data/{root}/train', transformations)
        testing_dataset = dataloader(f'data/{root}/test', transformations, mode = 'test')

    elif flag == 'classification':
        training_dataset = dataloader(root, transformations)
        testing_dataset = dataloader(root, transformations, mode = 'test')

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size//2, shuffle=False, num_workers=16)


    model = modelUtils.train_model(model= model, train_loader= training_loader, 
                                   test_loader=  testing_loader, loss_fn= loss_function, 
                                   eval_fn= eval_function, optimizer= optimizer, 
                                   epochs= num_epochs, output_dir= root,  scheduler= scheduler, device= device)

    modelUtils.validate_model(model= model, val_loader= testing_loader, loss_fn= loss_function, 
                              eval_fn= eval_function, output_dir = root, device= device,
                               target_layers= target_layers, flag = flag)

def run_yolo(dataset, 
                    class_names, 
                    model_path='yolov8n-obb.pt', 
                    epochs=100, 
                    batch_size=16, 
                    device_id=0, 
                    is_verbose=True, 
                    random_seed=0, 
                    perform_validation=True):


    project = f'{os.getcwd()}/data/{dataset}/logs/'

    data_yaml = {'path': f'{os.getcwd()}/data/{dataset}/', 'train': 'train/', 'val': 'test/', 'names': class_names}
    yaml_path = f'{os.getcwd()}/data/{dataset}/data.yaml'

    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file, sort_keys=False, default_flow_style=False)

    os.makedirs(project, exist_ok=True)

    model = YOLO(model_path)
    results = model.train(
        data = yaml_path,
        epochs = epochs,
        batch = batch_size,
        device = device_id,
        verbose = is_verbose,
        seed = random_seed,
        val = perform_validation,
        project = project
    )
    
    dataset_root=dataset
    modelUtils.bbox_prediction(dataset_root)

    print(results)