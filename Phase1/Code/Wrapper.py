#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2

# Add any python libraries here
import os
import argparse
import copy
from typing import Union
import matplotlib.pyplot as plt

CORNER_SCORE_THRESHOLD = 0.01
REGION_MAX_KERNEL = 5
CORNER_HARRIS_K = 0.04
DEBUG_LEVEL = 0

# returns a mask with same size as image
def region_maxima(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError(
            f'Kernal size must be odd. Currently is: {kernel_size}')
    output = np.zeros_like(image)
    for i in range(0, len(image), kernel_size):
        for j in range(0, len(image[0]), kernel_size):
            sub_mat = image[i:(i+kernel_size), j:(j+kernel_size)]
            max_val = sub_mat.max()
            row, col = np.where(sub_mat == max_val)
            if len(row) == 1:
                output[row+i, col+j] = 1
    return output


# could use a helper to decrease size of func. specifically the inner nested for loops.
def ANMS(corner_responses, num_best_corners) -> list[list[tuple[int,int]]]:
    # Gets the regional maximums and their coordinates
    n_best_list = []
    for response in corner_responses:
        output_mask = region_maxima(response, REGION_MAX_KERNEL)
        local_maxima = np.argwhere(output_mask)
        r = {}
        for i in local_maxima:
            pixel_coord = tuple(i)
            r[pixel_coord] = np.inf
            ED = np.inf
            for j in local_maxima:
                if response[j[0], j[1]] > response[i[0], i[1]]:
                    ED = (j[0] - i[0])**2 + (j[1]-i[1])**2
                if ED < r[pixel_coord]:
                    r[pixel_coord] = ED
        list = sorted(r.items(), key=lambda item: item[1])
        list.reverse()
        inf_removed = [x for x in list if x[1] != np.inf]
        n_best = []
        # for loop could be replaced by slice ':' operator
        for i in range(num_best_corners):
            n_best.append(inf_removed[i][0])
        n_best_list.append(n_best)
    return n_best_list

# relative path to dataset
def load_images(im_path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    for filename in os.listdir(im_path):
        full_image_path = im_path+filename
        image = cv2.imread(full_image_path, flags=flags)
        image_names.append(filename)
        images.append(image)
    return images, image_names

def write_images(images: Union[list[np.ndarray], np.ndarray] , image_names: Union[list[str], str]):
    if type(images) == list and type(image_names) == list:
        for image, name in zip(images, image_names):
            cv2.imwrite(name, image)
    elif type(images) == np.ndarray and type(image_names) == str:
        cv2.imwrite(name, image)
    else:
        raise(TypeError(f"Unsupported types recieved. Either list[np.ndarray], list[str] or np.ndarray, str. \n Given {type(images)}, {type(image_names)}"))

def write_anms_images(ANMS_scores, images_RGB, image_names, anms_out_path):
    im_list = []
    for coords_list, image in zip(ANMS_scores, images_RGB):
        im_copy = copy.deepcopy(image)
        for coords in coords_list:
            im_copy[coords[0], coords[1]] = [0, 0, 255]
        im_list.append(im_copy)

    write_images(im_list, [anms_out_path + "anms" + name for name in image_names])

def corner_viewer(corner_responses: Union[list[cv2.Mat], cv2.Mat], images_RGB: Union[list[np.ndarray], np.ndarray]) -> Union[list[np.ndarray], np.ndarray]:
    if type(corner_responses) == list and type(images_RGB) == list:
        corner_images = []
        for image, response in zip(images_RGB, corner_responses):
            image_cp = copy.deepcopy(image)
            image_cp[response > 0] = [0, 0, 255]
            corner_images.append(image_cp)
    elif type(corner_responses) == cv2.Mat and type(images_RGB) == np.ndarray:
        corner_images = copy.deepcopy(images_RGB)
        corner_images[corner_responses > 0] = [0, 0, 255]
    else:
        raise(TypeError(f"Unsupported types recieved. Either list[cv2.Mat], list[np.ndarray] or cv2.Mat, np.ndarray. \n Given {type(corner_responses)}, {type(images_RGB)}"))
    return corner_images

def generate_corner_responses(images_gray: list[np.ndarray], image_names: list[str]) -> tuple[list[np.ndarray], list[int]]:
    corner_responses = []
    corner_counts = []
    for img, name in zip(images_gray, image_names):
        response = cv2.cornerHarris(
            src=img, blockSize=2, ksize=3, k=CORNER_HARRIS_K)
        threshold = CORNER_SCORE_THRESHOLD * response.max()
        corner_image_mask = response > threshold
        count = np.sum(corner_image_mask)
        print(f"[{name}]: Found {count} corners ({round(100*count/(img.shape[0]*img.shape[1]), 3)}%)")
        if DEBUG_LEVEL > 0:
            plt.hist(response.flatten(), bins=1000)
            plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'x = {threshold}')
            plt.ylim([0,500])
            plt.show()
        response = np.multiply(np.uint8(corner_image_mask), response)
        corner_responses.append(response)
        corner_counts.append(count)
    return corner_responses, corner_counts

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImagePath', default='Phase1/Data/Train/Set1/',
                        help='Relative path to set of images you want to stitch together. Default:Phase1/Data/Train/Set1/')
    Parser.add_argument('--OutputPath', default='Phase1/Outputs/',
                        help='Output directory for all Phase 1 images. Default:Phase1/Outputs/')
    Parser.add_argument('--DebugLevel', type=int, default=0, help='increase debug verbosity with higher debug level')
    Args = Parser.parse_args()
    ImagePath = Args.ImagePath
    OutputPath = Args.OutputPath
    global DEBUG_LEVEL
    DEBUG_LEVEL = Args.DebugLevel
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    images_RGB, image_names = load_images(ImagePath, cv2.IMREAD_COLOR)
    images_gray, __ = load_images(ImagePath, cv2.IMREAD_GRAYSCALE)

    if not os.path.isdir(OutputPath):
        os.mkdir(OutputPath)

    """
        Corner Detection
        Save Corner detection output as corners.png
        """
    corner_responses, corner_count = generate_corner_responses(images_gray, image_names)

    corner_images = corner_viewer(corner_responses, images_RGB)
    corner_out_path = OutputPath+"Corners/"
    if not os.path.isdir(corner_out_path):
        os.mkdir(corner_out_path)

    write_images(corner_images, [corner_out_path + "corners" + name for name in image_names])

    """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
    ANMS_scores = ANMS(corner_responses, 500)
    anms_out_path = OutputPath+"anms/"
    if not os.path.isdir(anms_out_path):
        os.mkdir(anms_out_path)

    write_anms_images(ANMS_scores, images_RGB, image_names, anms_out_path)

    """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """

    """
        Feature Matching
        Save Feature Matching output as matching.png
        """

    """
        Refine: RANSAC, Estimate Homography
        """

    """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """


if __name__ == "__main__":
    main()
