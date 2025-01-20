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
import copy

# Add any python libraries here
import os
import argparse

CORNER_SCORE_THRESHOLD = 0.01

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


def ANMS(corner_score_img, num_best_corners):
    # Gets the regional maximums and their coordinates
    output_mask = region_maxima(corner_score_img, 5)
    # binary_matrix = corner_score_img > CORNER_SCORE_THRESHOLD*corner_score_img.max()
    local_maxima = np.argwhere(output_mask)
    r = {}
    for i in local_maxima:
        pixel_coord = tuple(i)
        r[pixel_coord] = np.inf
        ED = np.inf
        for j in local_maxima:
            if corner_score_img[j[0], j[1]] > corner_score_img[i[0], i[1]]:
                ED = (j[0] - i[0])**2 + (j[1]-i[1])**2
            if ED < r[pixel_coord]:
                r[pixel_coord] = ED
    list = sorted(r.items(), key=lambda item: item[1])
    list.reverse()
    inf_removed = [x for x in list if x[1] != np.inf]
    n_best = []
    for i in range(num_best_corners):
        n_best.append(inf_removed[i][0])
    return n_best

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


def write_images(images: list[np.ndarray], image_names: list[str]):
    for image, name in zip(images, image_names):
        print(type(image))
        print(name)
        cv2.imwrite(name, image)


def corner_viewer(corner_responses: list[cv2.Mat], images_RGB) -> list[np.ndarray]:
    corner_images = []
    for response, img in zip(corner_responses, images_RGB):
        image_cp = copy.deepcopy(img)
        image_cp[response > 0] = [0, 0, 255]
        corner_images.append(image_cp)
    return corner_images


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImagePath', default='Phase1/Data/Train/Set1/',
                        help='Relative path to set of images you want to stitch together. Default:Phase1/Data/Train/Set1/')
    Parser.add_argument('--OutputPath', default='Phase1/Outputs/',
                        help='Output directory for all Phase 1 images. Default:Phase1/Outputs/')
    Args = Parser.parse_args()
    ImagePath = Args.ImagePath
    OutputPath = Args.OutputPath
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """
    images_RGB, image_names = load_images(ImagePath, cv2.IMREAD_COLOR)
    images_gray, __ = load_images(ImagePath, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('image', images[0])
    # cv2.waitKey(delay=200)
    # closing all open windows

    if not os.path.isdir(OutputPath):
        os.mkdir(OutputPath)

    """
        Corner Detection
        Save Corner detection output as corners.png
        """
    corner_responses = []
    for i in range(0, len(images_gray)):
        corner_score = cv2.cornerHarris(
            src=images_gray[i], blockSize=2, ksize=3, k=0.04)

        corner_image_mask = corner_score > CORNER_SCORE_THRESHOLD * corner_score.max()
        corner_score = np.multiply(np.uint8(corner_image_mask), corner_score)
        corner_responses.append(corner_score)

    corner_images = corner_viewer(corner_responses, images_RGB)
    write_images(corner_images, [OutputPath + name for name in image_names])

    # TODO print num corners

    """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
    ANMS_scores = ANMS(corner_responses[2], 500)

    cccc = copy.deepcopy(images_RGB[2])
    # (4,4)
    for score_coord in ANMS_scores:
        cccc[score_coord[0], score_coord[1]] = [0, 0, 255]

    cv2.imwrite("ANMS_output.png", cccc)
    print(ANMS_scores)

    # for every corner, make

    # corner_images = corner_viewer([ANMS_scores], [images_RGB[2]])
    # cv2.imwrite("ANMS_output.png", corner_images)

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
