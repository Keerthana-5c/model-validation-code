from PIL import Image
import torch, os
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms

class NormalizeAndReplaceNaN(transforms.Normalize):
    def __init__(self):
        super(NormalizeAndReplaceNaN, self).__init__(mean=[0.0], std=[1.0])
    
    def __call__(self, tensor):
        pixels = tensor.numpy()
        mean = pixels.mean()
        std = pixels.std()
        pixels = (pixels - mean) / (std + 1e-8) 
        pixels_with_nan = np.isnan(pixels)
        pixels[pixels_with_nan] = 0
        return torch.from_numpy(pixels)
    
class CustomImageSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode = 'train'):
        self.transform = transform
        self.images = [os.path.join( os.path.join(root_dir, 'images'), img) for img in sorted(os.listdir( os.path.join(root_dir, 'images')))]
        self.masks = [os.path.join( os.path.join(root_dir, 'mask'), mask) for mask in sorted(os.listdir( os.path.join(root_dir, 'mask')))]
        self.mode = mode

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.mode == 'test':
            return image, mask, image_path.split('/')[-1].replace('.jpeg','')
        
        return image, mask


class CustomDatasetClassifier(Dataset):
    def __init__(self, root, transform = None, mode = 'train'):
        data = pd.read_csv(f"data/{root}/labels.csv")
        unique_labels = data['label'].unique()
        self.label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        self.images =  ([ (j['image_path'] , j['label']) for i,j in data[data['mode'] == mode].iterrows()])
        self.mode = mode
        self.transform = transform
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label =  self.images[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)

        if self.mode == 'test':
            return image,  self.label_to_index[label], image_path
        
        return image,  self.label_to_index[label]


        

