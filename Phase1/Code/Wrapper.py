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
import os
import argparse
import scipy
import scipy.ndimage
import diplib

# Add any python libraries here

def ANMS(corner_score_img, num_best_corners):
    # Gets the regional maximums and their coordinates 
    binary_matrix = corner_score_img>0.01*corner_score_img.max()
    local_maxima = np.argwhere(binary_matrix)
    r = {}
    for i in local_maxima:
        pixel_coord = tuple(i)
        r[pixel_coord] = np.inf
        ED = np.inf
        for j in local_maxima:
            if corner_score_img[j[0], j[1]] > corner_score_img[i[0], i[1]]:
                ED = (j[0] - i[1])**2 + (j[1]-i[1])**2
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
def load_images(im_path: str, flags: int=cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    for filename in os.listdir(im_path):
        full_image_path = im_path+filename 
        image = cv2.imread(full_image_path, flags=flags)
        image_names.append(filename)
        images.append(image)
    return images, image_names

def write_images(images:list[np.ndarray], image_names: list[str]):
    for image, name in zip(images, image_names):
        print(type(image))
        print(name)
        cv2.imwrite(name, image)

def corner_viewer(corner_responses:list[cv2.Mat], images_RGB) -> list[np.ndarray]:
    corner_images = []
    for response, img in zip(corner_responses, images_RGB):
        #threshhold re
        max_val = response.max()
        print(max_val)
        # .0112 max from one 
        # Threshold for an optimal value, it may vary depending on the image.
        img[response>0.01*response.max()]=[0,0,255]
        corner_images.append(img)
    return corner_images

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--ImagePath', default='Phase1/Data/Train/Set1/', help='Relative path to set of images you want to stitch together. Default:Phase1/Data/Train/Set1/')
    Parser.add_argument('--OutputPath', default='Phase1/Outputs/', help='Output directory for all Phase 1 images. Default:Phase1/Outputs/')
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
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # # dst = cv2.dilate(dst,None)
    # img[dst>0.01*dst.max()]=[0,0,255]  # Does the same as imregionalmax?
    # print(0.01*dst.max())
    # cv2.imwrite('corners.png',img)
    corner_responses = []
    for i in range(0, len(images_gray)):
        corner_responses.append(cv2.cornerHarris(src=images_gray[i], blockSize=2, ksize=3, k=0.04))

    corner_images = corner_viewer(corner_responses, images_RGB)
    write_images(corner_images, [OutputPath +name for name in image_names])

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
    ANMS_scores = ANMS(corner_responses[0], 10)
    
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
