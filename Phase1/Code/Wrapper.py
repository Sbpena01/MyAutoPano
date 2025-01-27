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
from scipy.optimize import least_squares
from Wrapper_Utilities import Point, Bounding_Box

CORNER_SCORE_THRESHOLD = 0.01
REGION_MAX_KERNEL = 5
CORNER_HARRIS_K = 0.04
NUM_BEST_CORNERS = 250 # from 500
N_MAX = 200 # from 100
TAU = 1e4 # from 1e4
INLIER_PERCENT_THRESHOLD = 0.75 # from 0.9
DISTANCE_RATIO_MAX = 0.7 # from 0.7
PANORAMA_WEIGHT = 0.8
WARPED_WEIGHT = 1.0 - PANORAMA_WEIGHT

DEBUG_LEVEL = 0


# TODO 

# Make fnding matches less sensitive

# Make main loop more robust to bad matches
#   It can be the case that the current panorama cannot currently stitch with selected RBG image, if so, we need to try the other images first, and revisit it afterwards

# Occasionally the pano doesnt contain the third image (probably due to a bad H matrix creation). debug this...

# Redo or modify blurring process, output is not suitable when H is not a good fit.

# Test whether refined H_matrix  is benefitting or harming us.

# Figure out Occasional Failure on Line 92


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

def ANMS(corner_response, num_best_corners) -> list[tuple[int, int]]:
    # Gets the regional maximums and their coordinates
    n_best = []
    output_mask = region_maxima(corner_response, REGION_MAX_KERNEL)
    local_maxima = np.argwhere(output_mask)
    r = {}
    for i in local_maxima:
        pixel_coord = Point(tuple(i))
        r[pixel_coord] = np.inf
        ED = np.inf
        for j in local_maxima:
            if corner_response[j[0], j[1]] > corner_response[i[0], i[1]]:
                ED = (j[0] - i[0])**2 + (j[1]-i[1])**2
            if ED < r[pixel_coord]:
                r[pixel_coord] = ED
    list = sorted(r.items(), key=lambda item: item[1])
    list.reverse()
    inf_removed = [x for x in list if x[1] != np.inf]
    n_best = []
    # for loop could be replaced by slice ':' operator
    # this fails with index out of range error. Probably if len(inf_removed) < num_best_corners
    for i in range(num_best_corners):
        n_best.append(inf_removed[i][0])
    return n_best

# relative path to dataset
def load_images(im_path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    filenames = os.listdir(im_path)
    filenames.sort(key=lambda x: int(x.split('.')[0]))  # Sort based on the number before the '.'
    for filename in filenames:
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
        im_mask = np.zeros((im_copy.shape[0], im_copy.shape[1]))
        for coords in coords_list:
            im_mask[coords.y, coords.x] = 1
        im_mask = cv2.dilate(im_mask, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,3)))
        im_copy[im_mask > 0] = [0, 0, 255]
        im_list.append(im_copy)

    write_images(im_list, [anms_out_path + "anms" +
                 name for name in image_names])


def grayscale_normalize(image: np.matrix) -> np.matrix:
    smallest = np.min(image)
    largest = np.max(image)

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

def generate_corner_response(images_gray: np.ndarray, image_name: str) -> tuple[np.ndarray, int]:
    response = cv2.cornerHarris(
        src=images_gray, blockSize=2, ksize=3, k=CORNER_HARRIS_K)
    threshold = CORNER_SCORE_THRESHOLD * response.max()
    corner_image_mask = response > threshold
    count = np.sum(corner_image_mask)
    print(f"[{image_name}]: Found {count} corners({round(100*count/(images_gray.shape[0]*images_gray.shape[1]), 3)} %)")
    if DEBUG_LEVEL > 0:
        plt.hist(response.flatten(), bins=1000)
        plt.axvline(x=threshold, color='red', linestyle='--',
                    linewidth=2, label=f'x = {threshold}')
        plt.ylim([0, 500])
        plt.show()
    response = np.multiply(np.uint8(corner_image_mask), response)
    return response, count


def get_subset(matrix: np.ndarray, subset_size: tuple):
    column_coordinates_float = np.linspace(0, matrix.shape[0], subset_size[0])
    column_coordinates_int = np.int8(column_coordinates_float)
    row_coordinates_float = np.linspace(0, matrix.shape[1], subset_size[1])
    row_coordinates_int = np.int8(row_coordinates_float)
    output = np.zeros(subset_size)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            output[i, j] = matrix[row_coordinates_int[i] -
                                  1, column_coordinates_int[j]-1]
    return output


def feature_descriptor(best_corners: list[Point], original_image: np.ndarray):
    padded_image = cv2.copyMakeBorder(
        original_image, 20, 20, 20, 20, cv2.BORDER_REFLECT)
    feature_dict = dict()
    for corner in best_corners:
        # must shift all coords over by 20 as we are expanding the image
        min_x = corner.x
        max_x = corner.x + 41
        min_y = corner.y
        max_y = corner.y + 41
        sub_region = padded_image[min_y:max_y, min_x:max_x]
        blurred_region = cv2.GaussianBlur(
            sub_region, ksize=(5, 5), sigmaX=1, sigmaY=1)
        sub_samble = get_subset(blurred_region, (8, 8))
        sub_sample_reshaped = np.reshape(sub_samble, -1)
        mean = np.mean(sub_sample_reshaped)
        std = np.std(sub_sample_reshaped)
        feature_dict[corner] = (sub_sample_reshaped - mean) / std
    return feature_dict


def feature_matcher(feature_dict_1: dict, feature_dict_2: dict, ratio_threshold=0.7):
    output_dictionary = dict()
    for image_1_point in feature_dict_1.keys():
        lowest_distance = np.inf
        second_lowest_distance = lowest_distance
        image_1_point_score = feature_dict_1[image_1_point]
        for image_2_point in feature_dict_2.keys():
            image_2_point_score = feature_dict_2[image_2_point]
            squared_distance = np.sum(
                (image_2_point_score - image_1_point_score) ** 2)
            if squared_distance < lowest_distance:
                second_lowest_distance = lowest_distance
                lowest_distance = squared_distance
                best_match = image_2_point
        distance_ratio = lowest_distance/second_lowest_distance
        if distance_ratio < DISTANCE_RATIO_MAX:
            output_dictionary[image_1_point] = best_match
    return output_dictionary


def write_matches(image1: np.ndarray, image2: np.ndarray, matches_dict: dict, match_outpath, image_pair_names):
    concat_image = cv2.hconcat([image1, image2])
    if DEBUG_LEVEL == 1:
        cv2.imshow('image', concat_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    for point1, point2 in matches_dict.items():
        cv2.circle(concat_image, (point1.x, point1.y), 1, (0, 0, 255), 2)
        cv2.circle(
            concat_image, (point2.x+image1.shape[1], point2.y), 1, (255, 0, 0), 2)
        cv2.line(concat_image, (point1.x, point1.y),
                 (point2.x+image1.shape[1], point2.y), (0, 255, 255), 1)

    name1_header = image_pair_names[0].split(".")[0]
    name2_header = image_pair_names[1].split(".")[0]
    cv2.imwrite(match_outpath+name1_header + "and" +
                name2_header + ".jpg", concat_image)

def generate_random_homography(matches_dict: dict):
    key_list = list(matches_dict.keys())
    points_1 = random.sample(key_list, k=4)
    points_2 = []
    for point in points_1:
        random_value = matches_dict[point]
        points_2.append(random_value)

    # Convert to numpy arrays
    # H = compute_homography(points_1, points_2)
    arr1 = np.array([point.to_numpy() for point in points_1], dtype=np.float32)
    arr2 = np.array([point.to_numpy() for point in points_2], dtype=np.float32)
    H = cv2.getPerspectiveTransform(arr1, arr2)
    if H is None:
        return generate_random_homography(matches_dict)
    return H


def compute_point_ssd(point1: Point, point2: Point, H: np.ndarray) -> float:
    point1_mat = np.array([
        [point1.x],
        [point1.y],
        [1]
    ])
    point2_mat = np.array([
        [point2.x],
        [point2.y],
        [1]
    ])
    point1_p_mat = np.matmul(H, point1_mat)
    return np.sum(np.square((point2_mat-point1_p_mat)))


def RANSAC(matches_dict: dict, n_max=N_MAX, tau=TAU):
    best_inlier_percent = 0.0
    best_homography = np.eye(3)
    best_inlier_dict = dict()
    for _ in range(n_max):
        H = generate_random_homography(matches_dict)
        inliers = dict()
        for point1, point2 in matches_dict.items():
            ssd = compute_point_ssd(point1, point2, H)
            if ssd < tau:
                inliers[point1] = point2
        inlier_percent = len(inliers)/len(matches_dict)
        # TODO: If this doesnt work, we need to figure out a better way to create the H from the set of inliers (least-square.. average translation...)
        if inlier_percent >= INLIER_PERCENT_THRESHOLD:
            best_inlier_dict = inliers
            best_homography = H
            break
        elif inlier_percent > best_inlier_percent:
            best_inlier_percent = inlier_percent
            best_inlier_dict = inliers
            best_homography = H

    print(best_homography)
    if DEBUG_LEVEL > 0:
        verbose = 2
    else:
        verbose = 0
    refined_homography_result = least_squares(homography_error_function, best_homography.flatten(), args=[
                                              best_inlier_dict], loss='cauchy', verbose=verbose)

    # return best_inlier_dict, np.reshape(refined_homography_result.x, (3,3))
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

    b = np.zeros((9, 1))
    b[8, 0] = 1
    if np.linalg.det(P) == 0:  # Matrix is singular
        return None
    H = np.linalg.solve(P, b)
    H = np.reshape(H, (3, 3))
    return H


def homography_error_function(h_guess, inliers_dict):
    h_guess = np.reshape(h_guess, (3, 3))
    total_error = 0
    for pi, pi_p in inliers_dict.items():
        total_error += compute_point_ssd(pi, pi_p, h_guess)
    return total_error

def warp_and_stitch(homography, image, panorama):

    corners = np.array([
                [0, 0],
                [image.shape[1], 0],
                [image.shape[1], image.shape[0]],
                [0, image.shape[0]]
            ], dtype=np.float32)

    # Convert corners to homogeneous coordinates
    corners = np.column_stack((corners, np.ones(corners.shape[0])))

    # Apply the homography matrix
    transformed_corners = np.dot(homography, corners.T) 

    # Normalize the points to convert back from homogeneous coordinates
    transformed_corners /= transformed_corners[2]

    # Extract x and y coordinates
    x_coords = transformed_corners[0]
    y_coords = transformed_corners[1]

    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    width = max(x_max, panorama.shape[1]) - min(0, x_min)
    height = max(y_max,panorama.shape[0]) - min(0, y_min)
    dsize = (width, height)

    # Offset to shift the result back into view if needed
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    offset_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # Update the homography
    H_offset = np.dot(offset_matrix, homography)
    # X increases horizontal axis, Y increases vertical axis
    warped_image = cv2.warpPerspective(image, M=H_offset, dsize=dsize)
    print(warped_image.shape)
    # cv2.imshow('warped_alone', warped_image)
    for y in range(warped_image.shape[0]):
        for x in range(warped_image.shape[1]):
            if y < panorama.shape[0] and x < panorama.shape[1]:
                warped_pixel = warped_image[y+offset_y,x+offset_x]
                if not np.equal(warped_pixel, np.array([0,0,0])).all():
                    # need to blend
                    warped_pixel = warped_pixel.astype(np.uint32) * WARPED_WEIGHT
                    panorama_pixel = panorama[y,x].astype(np.uint32) * PANORAMA_WEIGHT
                    new_pixel = np.add(warped_pixel, panorama_pixel).astype('uint8')
                else:
                    new_pixel = panorama[y,x]
                warped_image[y+offset_y,x+offset_x] = new_pixel
    cv2.imshow('pano', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return warped_image


def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100,
                        help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImagePath', default='Phase1/Data/Test/TestSet2/',
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

    """
    Read a set of images for Panorama stitching
    """
    images_RGB, image_names = load_images(ImagePath, cv2.IMREAD_COLOR)

    if not os.path.isdir(OutputPath):
        os.mkdir(OutputPath)

    homography_stack = []
    for idx in range(1, len(images_RGB)):
        image = images_RGB[idx-1]
        panorama = images_RGB[idx]
        
        # cv2.imshow("images", cv2.hconcat([image, panorama]))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        image_name = image_names.pop(0)
        image_to_greyscale = copy.deepcopy(image)
        greyscaled_image = cv2.cvtColor(image_to_greyscale, cv2.COLOR_BGR2GRAY)
        pano_to_greyscale = copy.deepcopy(panorama)
        greyscaled_pano = cv2.cvtColor(pano_to_greyscale, cv2.COLOR_BGR2GRAY)

        corner_response_image, count = generate_corner_response(
            greyscaled_image, image_name)
        corner_response_pano, count_pano = generate_corner_response(
            greyscaled_pano, "Pano")

        image_anms = ANMS(corner_response_image, NUM_BEST_CORNERS)
        pano_anms = ANMS(corner_response_pano, NUM_BEST_CORNERS)

        write_anms_images([image_anms, pano_anms], [image, panorama], ["image.jpg", "pano.jpg"], OutputPath+"anms/")

        image_feature_dict = feature_descriptor(image_anms, greyscaled_image)
        pano_feature_dict = feature_descriptor(pano_anms, greyscaled_pano)

        match_dict = feature_matcher(image_feature_dict, pano_feature_dict)

        # write_matches(image, panorama, match_dict,
        #           OutputPath + "Match/match", ("image.jpg", "pano.jpg"))

        inliers, homography = RANSAC(match_dict)
        print(f"[{image_name}] Found {len(inliers)} good matches ({round(100*len(inliers)/(len(match_dict)), 3)} %) ")

        # write_matches(image, panorama, inliers,
        #           OutputPath + "Match/RANSAC", ("image.jpg", "pano.jpg"))

        # panorama = warp_and_stitch(homography, image, panorama)
        homography_stack.append(homography)
        print()
        
    print(f"Found {len(homography_stack)} homography matricies from {len(images_RGB)} images.")

    image_idx = 1
    panorama = images_RGB[0]
    for H_idx in range(1, len(homography_stack)+1):
        H_list = homography_stack[0:H_idx]
        final_H = H_list[0]
        for H in H_list[1:]:
            final_H = np.matmul(final_H, H)
        panorama = warp_and_stitch(final_H, images_RGB[image_idx], panorama)
        image_idx += 1
        

    Pano_path = OutputPath+"Panoramas/"
    if not os.path.isdir(Pano_path):
        os.mkdir(Pano_path)
    cv2.imwrite(Pano_path+"Set3.jpg",panorama)

if __name__ == "__main__":
    main()
