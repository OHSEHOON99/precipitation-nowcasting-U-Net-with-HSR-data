import os
import torch
from torch.utils.data import Dataset
from utils import *


def list_files(src_dir, extension):
    return [os.path.join(src_dir, f) for f in os.listdir(src_dir)]


class TSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes an instance of the TSDataset class.

        Parameters:
        - data_dir (str): Path to the dataset directory.
        - transform (callable, optional): Optional transform to be applied to the data.
        """
        super().__init__()
        self.data_dir = str(data_dir)
        self.input_dirs = [os.path.join(data_dir, dir_name, 'input') for dir_name in os.listdir(data_dir)]
        self.label_dirs = [os.path.join(data_dir, dir_name, 'label') for dir_name in os.listdir(data_dir)]
        self.transform = transform

        # Process metadata
        metadata = self.data_dir.split("_")
        self.rfrate = float(metadata[1])
        self.rate = float(metadata[2])
        self.input_len = int(metadata[3])
        self.label_len = int(metadata[4])
        self.step = int(metadata[5][0])
        self.size = 256

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.input_dirs)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - dict: A dictionary containing inputs, labels, and corresponding image names.
        """
        input_path = self.input_dirs[idx]
        label_path = self.label_dirs[idx]
        input_images = sorted(list_files(input_path, '.png'), key=lambda x: int(os.path.basename(x)[:-4]))
        label_images = sorted(list_files(label_path, '.png'), key=lambda x: int(os.path.basename(x)[:-4]))

        # shape: (C, H, W)
        inputs = np.empty((self.input_len, self.size, self.size), dtype=np.float32)
        labels = np.empty((self.label_len, self.size, self.size), dtype=np.float32)

        # Load image data and store in arrays
        # Since dBZ values were shifted by +10 to avoid negative values, reverse normalization by adding 10
        for i in range(self.input_len):
            img = dBZ_to_pixel(png_to_dBZ(input_images[i]))
            inputs[i, :, :] = img

        for i in range(self.label_len):
            img = dBZ_to_pixel(png_to_dBZ(label_images[i]))
            labels[i, :, :] = img

        # Extract original image names
        input_image_names = [os.path.basename(path) for path in input_images]
        label_image_names = [os.path.basename(path) for path in label_images]

        data = {
            'input': inputs, 
            'label': labels,
            'input_names': input_image_names,  # Add list of original input image names
            'label_names': label_image_names   # Add list of original target image names
        }

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        """
        Converts numpy arrays in the data to PyTorch tensors.

        Parameters:
        - data (dict): Dictionary containing inputs and labels.

        Returns:
        - dict: Dictionary with inputs and labels converted to PyTorch tensors.
        """
        inputs, labels = data['input'], data['label']
        input_names, label_names = data['input_names'], data['label_names']  # Retain image names
        data = {
            'input': torch.from_numpy(inputs), 
            'label': torch.from_numpy(labels),
            'input_names': input_names,  # Keep image names as they are
            'label_names': label_names
        }
        return data


class AddSequenceDimension(object):
    def __call__(self, data):
        """
        Adds an extra dimension to the input and label tensors to represent the sequence dimension.

        Parameters:
        - data (dict): Dictionary containing inputs and labels.

        Returns:
        - dict: Dictionary with inputs and labels having an extra dimension.
        """
        inputs, labels = data['input'], data['label']

        # Convert input data from (C, H, W) -> (S, 1, H, W)
        # S represents the original C (channel) dimension, add a new dimension with size 1
        inputs = inputs.unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)
        labels = labels.unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)

        return {'input': inputs, 'label': labels}