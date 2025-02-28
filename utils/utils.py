import os
import numpy as np
import cv2


def list_files(src_dir, file_format):
    """
    Retrieve a sorted array of file paths in a directory and its subdirectories, filtered by a specified file format.

    Parameters
    ----------
    src_dir : str
        The source directory from which to start searching for files.
    file_format : str
        The desired file format or file extension to filter the files.

    Returns
    -------
    file_names : np.ndarray
        A sorted NumPy array containing the full paths of files that match the specified file format. (in this case, '.bin.gz')
    """
    # Initialize a list to store file paths
    file_names = []

    for (root, directories, files) in os.walk(src_dir):
        for file in files:
            if file.endswith(file_format):
                # Create the full path of the file and add it to the list
                img_dir = os.path.join(root, file)
                file_names.append(img_dir)

    # Sort the file paths in chronological order and convert them into a NumPy array
    file_names = np.array(sorted(file_names))

    return file_names


# Assuming the maximum value of dBZ is 70
# Very small values that are insignificant are shifted to -10: dBZ = -10 corresponds to 0.008 mm/hr
def dBZ_to_grayscale(dBZ):
    return 255 * ((dBZ + 10) / 70)


def grayscale_to_dBZ(img):
    return (img)*(70/255) - 10


# dBZ to rfrate : source(https://en.wikipedia.org/wiki/DBZ_(meteorology))
def dBZ_to_rfrate(dBZ):
    return (np.power(10, dBZ/10) / 200) ** (5/8)


def rfrate_to_dBZ(rfrate):
    return 10 * np.log10(200 * np.power(rfrate, 8 / 5))


def png_to_dBZ(img_path):
    """
    Convert a .png image to dBZ (decibel reflectivity) values using OpenCV.
    
    Args:
        img_path (str): Path to the .png image file.
        
    Returns:
        np.ndarray: The dBZ values as a NumPy array.
    """
    # Load the image as a 16-bit unsigned integer (uint16) in grayscale
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)

    # Convert stored value back to dBZ: dBZ = (stored_value - 1000) / 100
    dBZ = (image.astype(np.float32) - 1000) / 100.0
    
    return dBZ


def dBZ_to_pixel(dBZ):
    return (dBZ + 10) / 60.0


def pixel_to_dBZ(pixel):
    return pixel * 60.0 - 10