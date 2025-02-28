import os
import sys
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed
from datetime import datetime
import argparse

sys.path.append('../')

from utils import *


def extract_time_from_filename(filename):
    """
    Extract the timestamp from the filename assuming the format: 'YYYYMMDDHHMM.png'
    """
    base_name = os.path.basename(filename)
    timestamp_str = base_name[:12]  # Extract 'YYYYMMDDHHMM'
    return timestamp_str


def intensity_checker(img_names, rfrate, rate, size=256):
    """
    Filter images according to rainfall rate(=rfrate) and pixel rate(=rate) conditions.
    """
    L = len(img_names)
    print(f'Filter images according to rainfall rate conditions.\nWe have {L} images to be filtered')

    # Calculate the pixel threshold once outside the loop
    threshold = int(size * size * rate)

    # Initialize a NumPy array to store whether each image passes the filtering
    flag = np.zeros(L, dtype=bool)  # Use a NumPy array for faster operations

    for idx, img_path in tqdm(enumerate(img_names), total=L, ncols=100, dynamic_ncols=True):
        # Load the image and convert to dBZ
        dBZ = png_to_dBZ(img_path)

        # Convert dBZ to rainfall rate (rfrate)
        rfrate_array = dBZ_to_rfrate(dBZ)  # Assuming this function works with vectorized NumPy arrays

        # Vectorized intensity check: Count pixels that meet the rainfall rate condition
        if np.sum(rfrate_array >= rfrate) >= threshold:
            flag[idx] = True  # Set flag to True if condition is met

    print(f'Image filtering is finished!')

    return flag.tolist()  # Convert NumPy array back to list if necessary


def check_time_interval(timestamps, start_idx, length, step):
    """
    Check if the sequence follows the correct time interval based on the step size.
    This checks both hours and minutes.
    """
    for i in range(length - 1):
        current_time_str = timestamps[start_idx + i * step]
        next_time_str = timestamps[start_idx + (i + 1) * step]
        
        # Convert timestamp strings to datetime objects
        current_time = datetime.strptime(current_time_str, "%Y%m%d%H%M")
        next_time = datetime.strptime(next_time_str, "%Y%m%d%H%M")

        # Calculate the actual time difference in minutes
        actual_diff = (next_time - current_time).total_seconds() / 60
        
        # Expected time difference based on step
        expected_diff = step * 5  # For example, step = 2 means 10 minutes
        
        if actual_diff != expected_diff:
            return False
    return True


def generate_time_series_with_intensity_check(file_list, input_len, label_len, step, rfrate, rate, size=256):
    """
    Generate input and label index arrays ensuring correct time intervals and rainfall intensity check.
    The next sequence starts from the last used label's next value.
    """
    # Sort files by their extracted timestamp
    file_list_sorted = sorted(file_list, key=extract_time_from_filename)
    
    # Extract timestamps
    timestamps = [extract_time_from_filename(f) for f in file_list_sorted]
    
    # Perform intensity check on all files
    intensity_flag = intensity_checker(file_list_sorted, rfrate, rate, size)
    
    # Prepare to store valid indices
    valid_input_idx_array = []
    valid_label_idx_array = []
    total_frames = len(timestamps)

    # Initial starting point
    start_idx = 0

    # Continue until we reach the end of the list
    while start_idx + (input_len + label_len - 1) * step < total_frames:
        # Check the interval for input sequence
        input_start = start_idx  # Starting point of the input sequence
        label_start = input_start + input_len * step  # Starting point of the label sequence

        # Calculate the time series data for input and label sequences based on step
        if check_time_interval(timestamps, input_start, input_len, step) and check_time_interval(timestamps, label_start, label_len, step):
            # Check intensity for all input and label frames
            input_range = np.arange(input_start, input_start + input_len * step, step)
            label_range = np.arange(label_start, label_start + label_len * step, step)

            # Ensure that all the frames pass the intensity check
            if any(intensity_flag[idx] for idx in np.concatenate((input_range, label_range))):
                valid_input_idx_array.append(input_range.tolist())
                valid_label_idx_array.append(label_range.tolist())

        # The next sequence starts after the last label of the previous sequence
        start_idx = label_start + label_len * step

    return valid_input_idx_array, valid_label_idx_array


def process_sample(data_type, sample_idx, input_idx, label_idx, input_names, base_path):
    """
    Copy the files for input and label sequences to the respective directories.
    Each sequence will be saved under a sample_{n} folder with the original file names.
    """
    # Create directories for inputs and labels within sample_{n}
    input_dir = os.path.join(base_path, data_type, f'sample_{sample_idx}', 'input')
    label_dir = os.path.join(base_path, data_type, f'sample_{sample_idx}', 'label')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # Copy input images with their original filenames
    for idx in input_idx:
        original_filename = os.path.basename(input_names[idx])
        destination = os.path.join(input_dir, original_filename)
        shutil.copy(input_names[idx], destination)

    # Copy label images with their original filenames
    for idx in label_idx:
        original_filename = os.path.basename(input_names[idx])
        destination = os.path.join(label_dir, original_filename)
        shutil.copy(input_names[idx], destination)

def make_ts_dataset(data_dir, output_dir, rfrate, rate, input_len, label_len, step):
    """
    Generate a time-series dataset from cropped images based on specified conditions 
    and split into train, test, and validation sets.
    """
    # Initialize counters for train, test, and valid samples starting from 1
    n_train, n_test, n_valid = 1, 1, 1

    for year in range(2020, 2023+1):
        print(f'Starting data processing for the year {year}.')
        base_dir = os.path.join(data_dir, str(year))
        input_names = list_files(base_dir, file_format='.png')

        # Generate time-series indices with intensity check
        input_idx_array, label_idx_array = generate_time_series_with_intensity_check(input_names, input_len, label_len, step, rfrate, rate)

        num_data = len(input_idx_array)
        combined_data = list(zip(input_idx_array, label_idx_array))
        base_path = os.path.join(output_dir, f'dataset_{rfrate}_{rate}_{input_len}_{label_len}_{step}')

        os.makedirs(base_path, exist_ok=True)

        # Sequentially process the combined data, assigning to train, test, or valid sets
        for i, (input_idx, label_idx) in tqdm(enumerate(combined_data), total=num_data, desc='Generating Time-Series', ncols=100, dynamic_ncols=True):
            if i < int(0.8 * num_data):
                data_type = 'train'
                n = n_train
                n_train += 1  # Increment the train counter after assigning
            elif i < int(0.9 * num_data):
                data_type = 'test'
                n = n_test
                n_test += 1  # Increment the test counter after assigning
            else:
                data_type = 'valid'
                n = n_valid
                n_valid += 1  # Increment the valid counter after assigning

            # Create directories for the current sample (starting from 1)
            input_dir = os.path.join(base_path, data_type, f'sample_{n}', 'input')
            label_dir = os.path.join(base_path, data_type, f'sample_{n}', 'label')
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)

            # Copy input images with original filenames
            for j in range(input_len):
                idx = input_idx[j]
                img_path = input_names[idx]
                original_filename = os.path.basename(img_path)  # Keep the original filename
                shutil.copy(img_path, os.path.join(input_dir, original_filename))  # Copy with original filename

            # Copy label images with original filenames
            for j in range(label_len):
                idx = label_idx[j]
                label_path = input_names[idx]
                original_filename = os.path.basename(label_path)  # Keep the original filename
                shutil.copy(label_path, os.path.join(label_dir, original_filename))  # Copy with original filename

    print(f'Time-series dataset is saved to {base_path}\nTask Finished!')


def main():
    parser = argparse.ArgumentParser(description='Generate time series dataset from image data using specified rainfall intensity and 5-minute interval constraints.')
    
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='The input directory containing image files (e.g., PNG format) for time series generation.')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='The output directory where the processed time series dataset will be saved.')
    parser.add_argument('--rfrate', type=float, required=True, 
                        help='Rainfall rate threshold for intensity check. Images will only be included if a certain percentage of pixels exceed this rainfall intensity.')
    parser.add_argument('--rate', type=float, required=True, 
                        help='Pixel percentage threshold. Specifies the minimum percentage of pixels in the image that must exceed the rainfall rate threshold to be considered valid.')
    parser.add_argument('--input_len', type=int, required=True, 
                        help='The number of frames to use as input in each time series sequence.')
    parser.add_argument('--label_len', type=int, required=True, 
                        help='The number of frames to use as label/output in each time series sequence.')
    parser.add_argument('--step', type=int, required=True, 
                        help='The step size between consecutive frames in a sequence (for both input and label).')
    
    args = parser.parse_args()

    make_ts_dataset(args.input_dir, args.output_dir, args.rfrate, args.rate, args.input_len, args.label_len, args.step)

if __name__ == '__main__':
    main()