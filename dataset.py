import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from PIL import Image
import kagglehub
import os

class SignLanguageDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Args:
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Download (or get cached path) of the datamunge/sign-language-mnist dataset
        path = kagglehub.dataset_download("datamunge/sign-language-mnist")
        
        if split == 'train':
            csv_path = os.path.join(path, "sign_mnist_train.csv")
        else:
            # The test csv is inside another subfolder or in the root?
            # From list_dir it's sign_mnist_test/sign_mnist_test.csv or directly in root.
            # Wait, the list_dir showed:
            # {"name":"sign_mnist_test","isDir":true}
            # {"name":"sign_mnist_test.csv","sizeBytes":"21777485"}
            # So it's directly in the root of the dataset path.
            csv_path = os.path.join(path, "sign_mnist_test.csv")
            
        # Load CSV using pandas
        # The first column is 'label', the rest (784) are pixel values
        self.data = pd.read_csv(csv_path)
        self.labels = self.data['label'].values
        # Drop label column to get images
        self.images = self.data.drop('label', axis=1).values
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_pixels = self.images[idx]
        
        # Reshape to 28x28 and convert to uint8
        image_array = image_pixels.reshape(28, 28).astype(np.uint8)
        
        # Convert to PIL Image for the transforms to work properly
        image = Image.fromarray(image_array, mode='L')
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
