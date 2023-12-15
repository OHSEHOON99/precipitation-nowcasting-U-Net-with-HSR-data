import argparse
import os
import numpy as np
import random
import shutil

from PIL import Image
from tqdm import tqdm

# Adding the parent directory to the system path so that we can import modules from there
import sys
sys.path.append('../')

from utils import *

# Example usage
# python read_data.py --input_dir /home/u2018144071/사기설/data/cropped/ --output_dir /home/u2018144071/사기설/data/ --rfrate 5 --rate 0.05 --input_len 6 --label_len 1 --step 0


def intensity_checker(img_names, rfrate, rate, size=256):
    """
    Filter images according to rainfall rate(=rfrate) and pixel rate(=rate) conditions.

    Parameters
    ----------
    img_names : list
        List of image file paths.
    rfrate : float
        Rainfall rate threshold for filtering.
    rate : float
        Percentage threshold for filtering.
    size : int, optional
        Size of the images. Default is 256.

    Returns
    -------
    flag : list
        A list of boolean values indicating whether each image meets the intensity conditions.
    """
    L = len(img_names)
    print(f'Filter images according to rainfall rate conditions.\nWe have {L} images to be filtered')
    threshold = int(size * size * rate)
    flag = []

    for idx, img_path in tqdm(enumerate(img_names), total=L, ncols=100, dynamic_ncols=True):
        image = Image.open(img_path)
        image = np.array(image)

        # Convert pixel values back to dBZ values
        dBZ = grayscale_to_dBZ(image)

        # dBZ to rainfall rate
        rfrate_array = dBZ_to_rfrate(dBZ)

        # Intensity checker (256, 256): If the number of pixels with rfrate greater than or equal to the threshold is
        # higher than or equal to the overall threshold, set flag to True
        if np.sum(rfrate_array >= rfrate) >= threshold:
            flag.append(True)
        else:
            flag.append(False)

    print(f'Image filtering is finished!')

    return flag


def make_series_idx_array(flag, input_len, label_len, step):
    """
    Generate input and label index arrays based on specified conditions.

    Parameters
    ----------
    flag : list
        A list of boolean values indicating whether each image meets the intensity conditions.
    input_len : int
        Length of the input radar sequence.
    label_len : int
        Length of the label radar sequence.
        (eg. input_len = 6, label_len = 1 means predicting after 5 minutes with 6 radar images)
    step : int
        Steps between inputs and labels.
        (eg. step = 0 means predicting the next minute, step = 1 means predicting the next 5 minutes, etc.)

    Returns
    -------
    input_idx_array : list
        A list of lists containing input index sequences.
    label_idx_array : list
        A list of lists containing label index sequences.
    """
    input_idx_array = []
    label_idx_array = []

    L = len(flag)

    n = L - input_len - label_len - step + 1

    for i in range(n):
        # Check if the flag is True for all elements in the input and label sequences
        input_check = [flag[j] for j in range(i, i + input_len)]
        label = i + input_len + step
        label_check = [flag[j] for j in range(label, label + label_len)]

        if all(input_check + label_check):
            # If conditions are met, append the index sequences to the arrays
            input_series = [j for j in range(i, i + input_len)]
            label_series = [j for j in range(label, label + label_len)]
            input_idx_array.append(input_series)
            label_idx_array.append(label_series)

    return input_idx_array, label_idx_array


def make_ts_dataset(data_dir, output_dir, rfrate, rate, input_len, label_len, step):
    """
    Generate a time-series dataset from cropped images based on specified conditions and split it into train, test, and validation sets.

    Parameters
    ----------
    data_dir : str
        Directory containing cropped images in PNG format.
    output_dir : str
        Output directory to save the time-series dataset.
    rfrate : float
        Rainfall rate threshold for filtering.
    rate : float
        Percentage threshold for filtering.
    input_len : int
        Length of the input radar reflectivity image sequence.
    label_len : int
        Length of the label radar reflectivity image sequence.
    step : int
        Steps between inputs and labels.
    """
    n_train = 0
    n_test = 0
    n_valid = 0

    # we have 4 years of data
    for year in range(2020, 2023+1):
        print(f'Start with the data for the year {year}.')
        base_dir = os.path.join(data_dir, str(year))
        input_names = list_files(base_dir, file_format='.png')

        # Filtering images based on rainfall rate conditions
        flag = intensity_checker(input_names, rfrate, rate, size=256)
        
        # Generating time-series index arrays
        input_idx_array, label_idx_array = make_series_idx_array(flag, input_len, label_len, step)
        
        input_idx_array = np.sort(input_idx_array)
        label_idx_array = np.sort(label_idx_array)

        num_data = len(input_idx_array)

        # Randomly shuffling the data for better generalization
        combined_data = list(zip(input_idx_array, label_idx_array))
        # random.shuffle(combined_data)
        input_idx_array, label_idx_array = zip(*combined_data)

        base_path = os.path.join(output_dir, f'dataset_{rfrate}_{rate}_{input_len}_{label_len}_{step}')

        # To maintain the characteristics of time-series data, the ratio within each year is kept at 8:1:1.
        for i, (input_idx, label_idx) in tqdm(enumerate(combined_data), total=num_data, desc='Generating Time-Series', ncols=100, dynamic_ncols=True):
            if i < int(0.8 * num_data):
                data_type = 'train'
                n_train += 1
                n = n_train
            elif i < int(0.9 * num_data):
                data_type = 'test'
                n_test += 1
                n = n_test
            else:
                data_type = 'valid'
                n_valid += 1
                n = n_valid

            input_dir = os.path.join(base_path, data_type, f'sample_{n}', 'input')
            label_dir = os.path.join(base_path, data_type, f'sample_{n}', 'label')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # Copying input images to the appropriate directory
            for j in range(input_len):
                idx = input_idx[j]
                img_path = input_names[idx]
                shutil.copy(img_path, input_dir)

            # Copying label images to the appropriate directory
            for j in range(label_len):
                idx = label_idx[j]
                label_path = input_names[idx]
                shutil.copy(label_path, label_dir)

    print(f'Time-series dataset is saved to {base_path}\nTask Finished!')


def main():
    parser = argparse.ArgumentParser(description='Create a time series dataset with .h5py format')
    parser.add_argument('--input_dir', type=str, help='Input directory for time series data')
    parser.add_argument('--output_dir', type=str, help='Output path for the dataset')
    parser.add_argument('--rfrate', type=float, help='Rate for rainfall rate')
    parser.add_argument('--rate', type=float, help='Rate for pixel threshold')
    parser.add_argument('--input_len', type=int, help='Input length')
    parser.add_argument('--label_len', type=int, help='Label length')
    parser.add_argument('--step', type=int, help='Steps between inputs and labels')

    args = parser.parse_args()

    make_ts_dataset(args.input_dir, args.output_dir, args.rfrate,
                    args.rate, args.input_len, args.label_len, args.step)
    
if __name__ == '__main__':
    main()