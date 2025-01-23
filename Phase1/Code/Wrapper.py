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
import random
import hashlib
from typing import Union
import matplotlib.pyplot as plt

CORNER_SCORE_THRESHOLD = 0.01
REGION_MAX_KERNEL = 5
CORNER_HARRIS_K = 0.04
NUM_BEST_CORNERS = 500
N_MAX = 100
TAU = 1e4
INLIER_PERCENT_THRESHOLD = 0.9

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
def ANMS(corner_responses, num_best_corners) -> list[list[tuple[int, int]]]:
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

def write_images(images: Union[list[np.ndarray], np.ndarray], image_names: Union[list[str], str]):
    if type(images) == list and type(image_names) == list:
        for image, name in zip(images, image_names):
            cv2.imwrite(name, image)
    elif type(images) == np.ndarray and type(image_names) == str:
        cv2.imwrite(name, image)
    else:
        raise (TypeError(f"Unsupported types recieved. Either list[np.ndarray], list[str] or np.ndarray, str. \n Given {type(images)}, {type(image_names)}"))

def write_anms_images(ANMS_scores, images_RGB, image_names, anms_out_path):
    im_list = []
    for coords_list, image in zip(ANMS_scores, images_RGB):
        im_copy = copy.deepcopy(image)
        im_mask = np.zeros_like(im_copy)
        for coords in coords_list:
            im_mask[coords[0], coords[1]] = 1
        im_mask = cv2.dilate(im_mask, kernel=cv2.getStructuringElement(3))
        
    im_list.append(im_copy)

    write_images(im_list, [anms_out_path + "anms" +
                 name for name in image_names])


def grayscale_normalize(image: np.matrix) -> np.matrix:
    smallest = np.min(image)
    largest  = np.max(image)
    
    return np.uint8(255*(image - smallest) / (largest - smallest))

def write_feature_images(feature_dict: dict[tuple[int, int]], image_name, feature_outpath):
    # count keys, find a squareish number
    # 500 8x8s to stack
    if len(feature_dict.keys()) != NUM_BEST_CORNERS:
        return

    counter = 0
    FD_image = []  # np.zeros(((20*8), 25*8))
    FD_row = []

    for coord, feature in feature_dict.items():
        feature_square = copy.deepcopy(np.reshape(feature, ((8, 8))))
        feature_square = grayscale_normalize(feature_square)
        FD_row.append(feature_square)
        counter += 1
        if counter % 20 == 0:
            FD_image.append(np.hstack(FD_row))
            FD_row = []
    FD_image = np.vstack(FD_image)
    cv2.imwrite(feature_outpath + image_name, FD_image)
    return


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
        raise (TypeError(f"Unsupported types recieved. Either list[cv2.Mat], list[np.ndarray] or cv2.Mat, np.ndarray. \n Given {type(corner_responses)}, {type(images_RGB)}"))
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
        print(f"[{name}]: Found {count} corners ({
              round(100*count/(img.shape[0]*img.shape[1]), 3)}%)")
        if DEBUG_LEVEL > 0:
            plt.hist(response.flatten(), bins=1000)
            plt.axvline(x=threshold, color='red', linestyle='--',
                        linewidth=2, label=f'x = {threshold}')
            plt.ylim([0, 500])
            plt.show()
        response = np.multiply(np.uint8(corner_image_mask), response)
        corner_responses.append(response)
        corner_counts.append(count)
    return corner_responses, corner_counts

def get_subset(matrix: np.ndarray, subset_size:tuple):
    column_coordinates_float = np.linspace(0, matrix.shape[0], subset_size[0])
    column_coordinates_int = np.int8(column_coordinates_float)
    row_coordinates_float = np.linspace(0, matrix.shape[1], subset_size[1])
    row_coordinates_int = np.int8(row_coordinates_float)
    output = np.zeros(subset_size)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = matrix[row_coordinates_int[i]-1, column_coordinates_int[j]-1]
    return output

def feature_descriptor(best_corners: list, original_image: np.ndarray):
    padded_image = cv2.copyMakeBorder(original_image, 20, 20, 20, 20, cv2.BORDER_REFLECT)
    feature_dict = dict()
    for corner in best_corners:
        # must shift all coords over by 20 as we are expanding the image
        min_x = corner[0]
        max_x = corner[0] + 41
        min_y = corner[1]
        max_y = corner[1] + 41
        sub_region = padded_image[min_x:max_x, min_y:max_y]
        blurred_region = cv2.GaussianBlur(sub_region, ksize=(5,5), sigmaX=1, sigmaY=1)
        sub_samble = get_subset(blurred_region, (8, 8))
        # cv2.imshow('image', sub)
        # sub_sample = cv2.pyrDown(blurred_region, dstsize=(8,8)) # could be worth exploring in future
        
        sub_sample_reshaped = np.reshape(sub_samble, -1)
        mean = np.mean(sub_sample_reshaped)
        std = np.std(sub_sample_reshaped)
        feature_dict[(corner[0], corner[1])] = (sub_sample_reshaped - mean) / std
    return feature_dict

def feature_matcher(feature_dict_1: dict, feature_dict_2: dict, ratio_threshold=0.8):
    output_dictionary = dict()
    for image_1_point in feature_dict_1.keys():
        lowest_distance = np.inf
        second_lowest_distance = lowest_distance
        image_1_point_score = feature_dict_1[image_1_point]
        for image_2_point in feature_dict_2.keys():
            image_2_point_score = feature_dict_2[image_2_point]
            squared_distance = np.sum((image_2_point_score - image_1_point_score) ** 2)
            if squared_distance < lowest_distance:
                second_lowest_distance = lowest_distance
                lowest_distance = squared_distance
                best_match = image_2_point
        distance_ratio = lowest_distance/second_lowest_distance
        if distance_ratio < 0.7:
            output_dictionary[image_1_point] = best_match
    return output_dictionary

def write_matches(image1: np.ndarray, image2: np.ndarray, matches_dict: dict, match_outpath, image_pair_names):
    concat_image = cv2.hconcat([image1, image2])
    if DEBUG_LEVEL == 1:
        cv2.imshow('image', concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    for point1, point2 in matches_dict.items():
        cv2.circle(concat_image, (point1[1], point1[0]), 1, (0, 0, 255), 2)
        cv2.circle(
            concat_image, (point2[1]+image1.shape[1], point2[0]), 1, (255, 0, 0), 2)
        cv2.line(concat_image, (point1[1], point1[0]),
                 (point2[1]+image1.shape[1], point2[0]), (0, 255, 255), 1)

    name1_header = image_pair_names[0].split(".")[0]
    name2_header = image_pair_names[1].split(".")[0]
    cv2.imwrite(match_outpath+name1_header + "and" + name2_header + ".jpg", concat_image)


def RANSAC(matches_dict: dict, n_max=N_MAX, tau=TAU):
    key_list = list(matches_dict.keys())
    best_inlier_percent = 0.0
    best_homography = np.eye(3)
    best_inlier_dict = dict()
    points_1 = []
    points_2 = []
    for _ in range(n_max):
        points_1 = random.sample(key_list, k=4)
        points_2 = []
        for point in points_1:
            random_value = matches_dict[point]
            points_2.append(random_value)

        H = compute_homography(points_1, points_2)
        inliers = dict()
        for point1, point2 in matches_dict.items():
            point1_mat = np.array([
                [point1[1]],
                [point1[0]],
                [1]
            ])
            point2_mat = np.array([
                [point2[1]],
                [point2[0]],
                [1]
            ])
            point1_p_mat = np.matmul(H, point1_mat)
            ssd = np.sum(np.square((point2_mat-point1_p_mat)))
            if ssd < tau:
                inliers[point1] = point2
        inlier_percent = len(inliers)/len(key_list)
        # TODO: If this doesnt work, we need to figure out a better way to create the H from the set of inliers (least-square.. average translation...)
        if inlier_percent >= INLIER_PERCENT_THRESHOLD:
            best_inlier_percent = inlier_percent
            best_homography = H
            best_inlier_dict = inliers
            break
        elif inlier_percent > best_inlier_percent:
            best_inlier_percent = inlier_percent
            best_homography = H
            best_inlier_dict = inliers
    
    # Compute least square fit for final homographhy matrix
    # key_list = list(best_inlier_dict.keys())
    # points_1 = random.sample(key_list, k=4)
    # points_2 = []
    # for point in points_1:
    #     random_value = best_inlier_dict[point]
    #     points_2.append(random_value)
    
    # best_homography = compute_homography(points_1, points_2)
    # best_inlier_dict = dict()
    # for point1, point2 in matches_dict.items():
    #     point1_mat = np.array([
    #         [point1[1]],
    #         [point1[0]],
    #         [1]
    #     ])
    #     point2_mat = np.array([
    #         [point2[1]],
    #         [point2[0]],
    #         [1]
    #     ])

    # for point1, point2 in matches_dict.items():
    #         point1_mat = np.array([
    #             [point1[1]],
    #             [point1[0]],
    #             [1]
    #         ])
    #         point2_mat = np.array([
    #             [point2[1]],
    #             [point2[0]],
    #             [1]
    #         ])
    #         point1_p_mat = np.matmul(H, point1_mat)
    #         ssd = np.sum(np.square((point2_mat-point1_p_mat)))
    #         if ssd < tau:
    #             best_inlier_dict[point1] = point2
    
    return best_inlier_dict, best_homography

def compute_homography(points_1, points_2):
    p1, p2, p3, p4 = points_1
    p1_p, p2_p, p3_p, p4_p = points_2
    
    # set up PH matrix
    P = np.array([
        [-p1[1], -p1[0], -1, 0, 0, 0, p1[1]*p1_p[1], p1[0]*p1_p[1], p1_p[1]],
        [0, 0, 0, -p1[1], -p1[0], -1, p1[1]*p1_p[0], p1[0]*p1_p[0], p1_p[0]],
        [-p2[1], -p2[0], -1, 0, 0, 0, p2[1]*p2_p[1], p2[0]*p2_p[1], p2_p[1]],
        [0, 0, 0, -p2[1], -p2[0], -1, p2[1]*p2_p[0], p2[0]*p2_p[0], p2_p[0]],
        [-p3[1], -p3[0], -1, 0, 0, 0, p3[1]*p3_p[1], p3[0]*p3_p[1], p3_p[1]],
        [0, 0, 0, -p3[1], -p3[0], -1, p3[1]*p3_p[0], p3[0]*p3_p[0], p3_p[0]],
        [-p4[1], -p4[0], -1, 0, 0, 0, p4[1]*p4_p[1], p4[0]*p4_p[1], p4_p[1]],
        [0, 0, 0, -p4[1], -p4[0], -1, p4[1]*p4_p[0], p4[0]*p4_p[0], p4_p[0]],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])

    b = np.zeros((9,1))
    b[8,0] = 1
    H = np.linalg.solve(P,b)
    H = np.reshape(H, (3,3))
    return H

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImagePath', default='Phase1/Data/Train/Set1/',
                        help='Relative path to set of images you want to stitch together. Default:Phase1/Data/Train/Set1/')
    Parser.add_argument('--OutputPath', default='Phase1/Outputs/',
                        help='Output directory for all Phase 1 images. Default:Phase1/Outputs/')
    Parser.add_argument('--DebugLevel', type=int, default=0,
                        help='increase debug verbosity with higher debug level')
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
    corner_responses, corner_count = generate_corner_responses(
        images_gray, image_names)

    corner_images = corner_viewer(corner_responses, images_RGB)
    corner_out_path = OutputPath+"Corners/"
    if not os.path.isdir(corner_out_path):
        os.mkdir(corner_out_path)

    write_images(corner_images, [corner_out_path +
                 "corners" + name for name in image_names])

    """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
    ANMS_output = ANMS(corner_responses, NUM_BEST_CORNERS)
    anms_out_path = OutputPath+"anms/"
    if not os.path.isdir(anms_out_path):
        os.mkdir(anms_out_path)

    # write_anms_images(ANMS_output, images_RGB, image_names, anms_out_path)

    """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
    """
    feature_dict0 = feature_descriptor(ANMS_output[0], images_gray[0])
    feature_dict1 = feature_descriptor(ANMS_output[1], images_gray[1])

    fd_outpath = OutputPath + "FD/"
    if not os.path.isdir(fd_outpath):
        os.mkdir(fd_outpath)
    write_feature_images(feature_dict0, image_names[0], fd_outpath + "FD")

    # cv2.imshow("Original", images_RGB[0])
    # cv2.imshow("Padded", cv2.copyMakeBorder(images_RGB[0], 20, 20, 20, 20, cv2.BORDER_REFLECT))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
        Feature Matching
        Save Feature Matching output as matching.png
        """
    match_outpath = OutputPath + "Match/"
    if not os.path.isdir(match_outpath):
        os.mkdir(match_outpath)
    match_dict = feature_matcher(feature_dict0, feature_dict1)
    write_matches(images_RGB[0], images_RGB[1], match_dict, match_outpath + "Match", (image_names[0], image_names[1]))

    """
        Refine: RANSAC, Estimate Homography
    """

    good_matches, homography = RANSAC(match_dict)
    print(f"[{image_names[0]}] Found {len(good_matches)} good matches ({round(100*len(good_matches)/(len(match_dict)), 3)} %) ")

    write_matches(images_RGB[0], images_RGB[1], good_matches, match_outpath+ "RANSAC", (image_names[0], image_names[1]))

    """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """


if __name__ == "__main__":
    main()
