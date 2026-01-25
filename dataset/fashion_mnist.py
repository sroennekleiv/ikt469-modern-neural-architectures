import torch
import numpy as np

from keras.datasets import fashion_mnist
from torch.utils.data import TensorDataset, DataLoader

from data_preparation.preprocessing import Preprocessor
from data_preparation.split_dataset import SplitDataset

class FashionMNISTDataset:
    def __init__(self, batch_size=64, augment=True, size=(28, 28), validation_split=0.2):
        # Load raw data (no preprocessing)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        self.input_shape = self.x_train.shape[1:]  # (28, 28)
        self.num_classes = 10
        
        self.augment = augment
        self.size = size
        self.validation_split = validation_split
        
        self.batch_size = batch_size
        
        # Preprocess images
        self.preprocessor = Preprocessor(lower=False, size=self.size)
        self.x_train = self.preprocessor.preprocess_images(self.x_train, augment=True)
        self.x_test = self.preprocessor.preprocess_images(self.x_test, augment=False)
        
        # Convert to numpy arrays
        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)

        # Split training data into training and validation sets
        splitter = SplitDataset(val_fraction=self.validation_split, seed=42)
        self.train_x, self.train_y, self.val_x, self.val_y = splitter.split(self.x_train, self.y_train)
        
        # Convert to tensors (NCHW format for CNNs)
        self.train_x = torch.tensor(self.train_x, dtype=torch.float32).permute(0, 3, 1, 2) # [batch, 1, 28, 28]
        self.val_x = torch.tensor(self.val_x, dtype=torch.float32).permute(0, 3, 1, 2)     # [batch, 1, 28, 28]
        self.x_test = torch.tensor(self.x_test, dtype=torch.float32).permute(0, 3, 1, 2)   # [batch, 1, 28, 28]
        
        # Create TensorDatasets
        self.train_dataset = TensorDataset(self.train_x, torch.tensor(self.train_y, dtype=torch.long))
        self.val_dataset = TensorDataset(self.val_x, torch.tensor(self.val_y, dtype=torch.long))
        self.test_dataset = TensorDataset(self.x_test, torch.tensor(self.y_test, dtype=torch.long))

        # Build DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        

    def get_data(self):
        return (self.train_loader, self.val_loader, self.test_loader)

    def get_input_shape(self):
        return self.input_shape

    def get_num_classes(self):
        return self.num_classes
    
    def get_target_names(self):
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']