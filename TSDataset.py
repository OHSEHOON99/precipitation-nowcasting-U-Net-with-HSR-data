import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from utils import *


class TSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initializes an instance of the TSDataset class.

        Parameters:
        - data_dir (str): Directory containing the dataset in the format "Dataset_{rfrate}_{rate}_{input_len}_{label_len}_{step}".
        - transform (callable, optional): A function/transform to apply to the data. Default is None.
        """
        # data_dir format: Dataset_{rfrate}_{rate}_{input_len}_{label_len}_{step}
        self.data_dir = data_dir
        metadata = data_dir.split("_")

        self.rfrate = float(metadata[1])
        self.rate = float(metadata[2])
        self.input_len = int(metadata[3])
        self.label_len = int(metadata[4])
        self.step = int(metadata[5][0])
        self.size = 256

        # Initialize data arrays
        # self.scaler = MinMaxScaler(): Changed to simple normalization by dividing by 255.0
        self.input_array, self.label_array = self.load_data()
        self.num_samples = len(self.input_array)
        self.transform = transform

        # Other scaler options can be used.

    def load_data(self):
        sample_array = [os.path.join(self.data_dir, sample_path) for sample_path in os.listdir(self.data_dir)]
        input_array = []
        label_array = []

        for idx, sample_dir in enumerate(sample_array):
            input_dir = os.path.join(sample_dir, 'input')
            label_dir = os.path.join(sample_dir, 'label')

            input_imgs = os.listdir(input_dir)
            label_imgs = os.listdir(label_dir)

            input_data = np.empty((self.size, self.size, self.input_len), dtype=np.float32)
            label_data = np.empty((self.size, self.size, self.label_len), dtype=np.float32)

            # Read and store input images as arrays
            for j, img_path in enumerate(input_imgs):
                img = Image.open(os.path.join(input_dir, img_path))

                # Min-max normalization -> Modified to normalization by dividing by 255.0
                img_array = np.array(img, dtype=np.float32)
                # img_normalized = self.scaler.fit_transform(img_array.reshape(-1, 1)).reshape(img_array.shape)
                img_normalized = img_array / 255.0
                input_data[..., j] = img_normalized

            # Read and store label images as arrays
            for j, img_path in enumerate(label_imgs):
                img = Image.open(os.path.join(label_dir, img_path))

                # Min-max normalization -> Modified to normalization by dividing by 255.0
                img_array = np.array(img, dtype=np.float32)
                # img_normalized = self.scaler.fit_transform(img_array.reshape(-1, 1)).reshape(img_array.shape)
                img_normalized = img_array / 255.0
                label_data[..., j] = img_normalized

            input_array.append(input_data)
            label_array.append(label_data)

        input_array = np.array(input_array)
        label_array = np.array(label_array)

        return input_array, label_array

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        inputs = self.input_array[idx]  # shape = (height, width, input_len)
        labels = self.label_array[idx]  # shape = (height, width, label_len)

        data = {'input': inputs, 'label': labels}

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # Numpy and tensor arrays have different dimension orders:
        # Numpy: (height, width, channels)
        # Tensor: (channels, height, width)
        # Transpose to match the tensor dimension order
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data