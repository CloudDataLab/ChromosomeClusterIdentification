from PIL import Image
import cv2
import copy
from skimage import measure
import os
from warnings import simplefilter
import pandas
from glob import glob
import numpy as np
simplefilter(action='ignore', category=Warning)

img_path = '../dataset/images'
feature_file_path = '../dataset/chromosome_geometric_features.csv'
min_area = 260

file_name_list = []
solidity_list = []
area_list = []
perimeter_list = []
equivalent_diameter_list = []
eccentricity_list = []
labels = []
bbox_area_list = []
extent_list = []
major_axis_length_list = []
min_axis_length_list = []
min_major_axis_ratio_list = []
convex_area_list = []

split = '\\'


def get_chromosome_slices(image_array, connectivity=8):
    img1 = copy.deepcopy(image_array)
    img1[img1 != 255] = 0
    ret, thresh1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)
    # ret, thresh1 = cv2.threshold(img1, 250, 255, cv2.THRESH_BINARY)
    thresh1_1 = thresh1[:, :]
    thresh1_1[thresh1_1 == 0] = 2
    thresh1_1[thresh1_1 == 255] = 0
    slice_value = copy.deepcopy(thresh1_1)
    slices = measure.label(slice_value, neighbors=connectivity)
    return slices


label_files = glob(os.path.join(img_path, '*/*/*.TIF'))
for index, file_path in enumerate(label_files):
    file_name_list.append(file_path)
    if file_path.split(split)[2].startswith('chromosome'):
        labels.append(0)
    else:
        labels.append(1)
    # image_array = np.array()
    img = Image.open(file_path)
    image_array = np.array(img)
    slices = get_chromosome_slices(image_array, connectivity=8)
    for region in measure.regionprops(slices):
        solidity_list.append(region.solidity)
        perimeter_list.append(region.perimeter)
        equivalent_diameter_list.append(region.equivalent_diameter)
        eccentricity_list.append(region.eccentricity)
        bbox_area_list.append(region.bbox_area)
        area_list.append(region.area)
        extent_list.append(region.extent)
        major_axis_length_list.append(region.major_axis_length)
        min_axis_length_list.append(region.minor_axis_length)
        min_major_axis_ratio_list.append(region.minor_axis_length/region.major_axis_length)
        convex_area_list.append(region.convex_area)
        break
data_frame = pandas.DataFrame({
    'file_name' : file_name_list,
    'solidity' : solidity_list,
    'perimeter': perimeter_list,
    'equivalent_diameter' : equivalent_diameter_list,
    'eccentricity' : eccentricity_list,
    'bbox_area' : bbox_area_list,
    'area' : area_list,
    'extent' : extent_list,
    'major_axis_length' : major_axis_length_list,
    'min_axis_length' : min_axis_length_list,
    'min_major_axis_ratio' : min_major_axis_ratio_list,
    'convex_area' : convex_area_list,
    'type' : labels
})
data_frame.to_csv(feature_file_path, index=False, sep=',')




