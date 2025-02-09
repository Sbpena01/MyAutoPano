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
import torch
import os
import random
import Utilities
from Network.Network import HomographyModel
from DataGeneration import load_images, warp_bb, generate_patch, Bounding_Box, Point
import copy
import time
# Add any python libraries here


def main():
    # load model 
    sup_model = HomographyModel('Sup')
    # unsup_model = HomographyModel('Unsup')
    sup_model.eval()
    # unsup_model.eval()

    sup_model.load_state_dict(torch.load("Phase2/Checkpoints/49modelcpu.ckpt", weights_only=True), strict=False)
    # unsup_model.load_state_dict(torch.load("Phase2/Checkpoints_Unsup/0modelcpu.ckpt", weights_only=True), strict=False)



    """
    Read a set of images for Panorama stitching
    """

    images, im_names = load_images("Phase2/Data/Phase2Pano/trees/", -1)

    for i in range(len(images)):
        images[i] = cv2.resize(images[i], dsize=(int(images[i].shape[1]*.25),int(images[i].shape[0]*.25)))

    print(im_names)
    image_patch_pair = []
    im_shape = images[0].shape
    im_bb = Bounding_Box(
            Point((0,0)),
            Point((im_shape[1], 0)),
            Point((0, im_shape[0])),
            Point((im_shape[1], im_shape[0]))
        )

    for i in range(1,len(images)):
        # assumes that both images have the same dimensions 
        rand_patch = generate_patch(images[i])
        rand_patch_im = images[i-1][rand_patch.tl.y:rand_patch.bl.y, rand_patch.tl.x:rand_patch.br.x]
        rand_patch_im2 = images[i][rand_patch.tl.y:rand_patch.bl.y, rand_patch.tl.x:rand_patch.br.x]
        image_patch_pair.append(np.array([rand_patch_im, rand_patch_im2]))


    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    total_runtime = 0
    # corners = np.array([[0,0], [0, 128], [128,0], [128,128]])
    corners = im_bb.get_points_np()
    h_list = []
    for patch_pair in image_patch_pair:
        start_time = time.perf_counter_ns()
        with torch.no_grad():
            sup_output = sup_model(torch.tensor([patch_pair], dtype=torch.float32))
        runtime = time.perf_counter_ns() - start_time
        total_runtime += runtime
        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        sup_output = sup_output.numpy().reshape((4,2))
        homography_estim_sup, __ = cv2.findHomography(corners, corners+sup_output)
        h_list.append(homography_estim_sup)

    average_runtime = 1e-9*(total_runtime / len(h_list))
    print(f"Average Runtime for Supervised Model: {round(average_runtime,5)} seconds")

    # concat each image onto the original 
    h_pano = np.array([[1,0,0], [0,1,0], [0,0,1]])
    pano = images.pop(0)
    for homography,im in zip(h_list, images):
        h_pano = np.dot(h_pano, homography)
        
        dsize, h_offset = Utilities.calculate_dsize(im, h_pano)
        h_pano = np.dot(h_offset, h_pano)
        pano = Utilities.warp_and_stitch(h_pano, im, pano)
    cv2.imwrite("Panorama.jpg", pano)

if __name__ == "__main__":
    main()
