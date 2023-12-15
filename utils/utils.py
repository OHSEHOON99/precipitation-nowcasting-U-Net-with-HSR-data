import datetime
import os
import math
import numpy as np

from IPython.display import Image as Img
from IPython.display import display
from PIL import Image
from pyproj import Proj, transform


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

def dBZ_to_grayscale(dBZ):
    return 255*((dBZ + 10)/70) + 0.5


def grayscale_to_dBZ(img):
    return (img - 0.5)*(70/255) - 10


def dBZ_to_rfrate(dBZ):
    # dBZ to rfrate : source(https://en.wikipedia.org/wiki/DBZ_(meteorology))
    return (np.power(10, dBZ/10) / 200) ** (5/8)


def rfrate_to_dBZ(rfrate):
    return 10*math.log10(200 * rfrate ** (8 / 5))


def extract_date_from_file_name(file_path):
    file_name = os.path.basename(file_path)
    file_date = file_name.split('.')[0]
    file_date = datetime.strptime(file_date, "%Y%m%d%H%M")

    return file_date

def generate_gif(img_path, save_path):
    img_list = os.listdir(img_path)

    # 날짜 순서대로 sort
    img_list.sort()
    # print(img_list)
    img_list = [img_path + '/' + x for x in img_list]
    images = [Image.open(x) for x in img_list]
    
    im = images[0]
    im.save(save_path, save_all=True, append_images=images[1:],loop=0xff, duration=200)
    # loop 반복 횟수
    # duration 프레임 전환 속도 (500 = 0.5초)
    return Img(url=save_path)

def lonlat_to_xy(lon, lat, epsg):
    # use pyproj to convert lat/lon to x/y coordinates in meters
    # source: https://gis.stackexchange.com/questions/78838/converting-projected-coordinates-to-lat-lon-using-python
    
    proj_latlon = Proj(init='epsg:4326')   # WGS84
    proj_xy = Proj(init=epsg) # korea 2000 / central belt
    # Convert lat/lon to x/y
    x, y = transform(proj_latlon, proj_xy, lon, lat)

    return x, y
