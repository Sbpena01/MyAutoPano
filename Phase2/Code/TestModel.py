import torch
import cv2
import os
import random
import Utilities
from Network.Network import HomographyModel
import DataGeneration
import numpy as np
import copy

def main():
    # load model 
    device = torch.device('cuda')
    device_cpu = torch.device('cpu')
    sup_model = HomographyModel('Sup')
    unsup_model = HomographyModel('Unsup')
    # model.to(device)
    sup_model.eval()
    unsup_model.eval()

    # Checkpoint = torch.load("Phase2/Checkpoints/49modelcpu.ckpt", map_location=device_cpu)
    # model.load_state_dict(Checkpoint["model_state_dict"])
    sup_model.load_state_dict(torch.load("Phase2/Checkpoints/49modelcpu.ckpt", weights_only=True), strict=False)
    unsup_model.load_state_dict(torch.load("Phase2/Checkpoints_Unsup/0modelcpu.ckpt", weights_only=True), strict=False)
    
    idx = 1
    image = cv2.imread(f"Phase2/Data/Val/{idx}.jpg", cv2.IMREAD_GRAYSCALE)
    original_patch = DataGeneration.generate_patch(image)
    perturbed_patch = DataGeneration.perterbate_patch(original_patch)
    homography = cv2.getPerspectiveTransform(original_patch.get_points_np(), perturbed_patch.get_points_np())
    homography_inv = np.linalg.inv(homography)
    dsize, offset_matrix = Utilities.calculate_dsize(image, homography_inv)
    H_offset = np.dot(offset_matrix, homography_inv)

    warped_image = cv2.warpPerspective(image, M=H_offset, dsize=dsize)
    warped_patch = warped_image[int(original_patch.tl.y+offset_matrix[1,2]):int(original_patch.bl.y+offset_matrix[1,2]),
                                        int(original_patch.tl.x+offset_matrix[0, 2]):int(original_patch.br.x+offset_matrix[0,2])]

    DataGeneration.display_bounding_boxes('unwarped image w/ Origin and GT', image, (original_patch, perturbed_patch))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    unwarped_patch = image[original_patch.tl.y:original_patch.bl.y, original_patch.tl.x:original_patch.br.x]
    # GROUND TRUTH GENERATION ENDS HERE


    patches_model_Sup = torch.from_numpy(np.array([[unwarped_patch, warped_patch]], dtype=np.float32))
    patches_model_Unsup = copy.deepcopy(patches_model_Sup)
    print(patches_model_Sup.shape)
    # patches_model = patches_model.to(device)
    with torch.no_grad():
        sup_output = sup_model(patches_model_Sup)
        __, unsup_output = unsup_model(patches_model_Unsup, torch.tensor(original_patch.get_points_np().reshape((1,8))), idx, is_train=False)  # First 
    
    # gives us a h4pt...
    # translate this into a 3x3 homography...
    H_4pt = perturbed_patch.get_points_np() - original_patch.get_points_np()
    print(H_4pt)
    sup_output = sup_output.numpy().reshape((4,2))
    print(sup_output)

    # Supervised output:
    
    
    # homography_estim_sup = Utilities.compute_homography(torch.tensor(original_patch.get_points_np()), torch.tensor(original_patch.get_points_np()+sup_output))
    homography_estim_sup, __ = cv2.findHomography(original_patch.get_points_np(), original_patch.get_points_np()+sup_output)

    print(homography_estim_sup)
    # homography_estim_sup3 = Utilities.find_homography(original_patch.get_points_np(), original_patch.get_points_np()+sup_output)
    # print(homography_estim_sup)
    # print("\n")
    # print(homography_estim_sup2)
    # print("\n")
    # print(homography_estim_sup3)
    # print("\n")
    # exit(0)



    
    homography_estim_inv_sup = np.linalg.inv(homography_estim_sup)
    dsize_estim_sup, offset_matrix_estim_sup = Utilities.calculate_dsize(image, homography_estim_inv_sup)
    sup_homography_estim_offset = np.dot(offset_matrix_estim_sup, homography_estim_inv_sup)
    warped_image_sup = cv2.warpPerspective(image, sup_homography_estim_offset, dsize=dsize_estim_sup)
    # print(homography)
    # print("\n")
    # print(homography_estim_sup)
 
    sup_warped_bb_estim = DataGeneration.warp_bb(original_patch, homography_estim_sup)

    # Unsupervised output:
    unsup_output = unsup_output.numpy().reshape((4,2))
    homography_estim_unsup = Utilities.compute_homography(torch.tensor(original_patch.get_points_np()), torch.tensor(original_patch.get_points_np()+unsup_output))
    homography_estim_inv_unsup = np.linalg.inv(homography_estim_unsup)
    dsize_estim_unsup, offset_matrix_estim_unsup = Utilities.calculate_dsize(image, homography_estim_inv_unsup)
    unsup_homography_estim_offset = np.dot(offset_matrix_estim_unsup, homography_estim_inv_unsup)
    warped_image_unsup = cv2.warpPerspective(image, unsup_homography_estim_offset, dsize=dsize_estim_unsup)
    # print(homography)
    # print("\n")
    # print(homography_estim_unsup)
 
    unsup_warped_bb_estim = DataGeneration.warp_bb(original_patch, homography_estim_unsup)

    H_offset_2 = np.dot(offset_matrix_estim_sup, homography_inv)

    # H_output_inv = np.linalg.inv(H_output)
    # # H_output_forward = np.dot(offset_matrix_output, H_output)
    # H_output_inv = np.dot(offset_matrix_output, H_output_inv)
    DataGeneration.display_bounding_boxes('Unwarped image w/ GT and Estim from Supervised Model', image, (perturbed_patch, sup_warped_bb_estim))
    
    DataGeneration.display_bounding_boxes('Warped Image w/ GT and Estim from Supervised Model', warped_image_sup, 
                                          (DataGeneration.warp_bb(perturbed_patch, H_offset_2), 
                                           DataGeneration.warp_bb(sup_warped_bb_estim, sup_homography_estim_offset)))
    

    DataGeneration.display_bounding_boxes('Unwarped image w/ GT and Estim from Unsupervised Model', image, (perturbed_patch,unsup_warped_bb_estim))
    
    DataGeneration.display_bounding_boxes('Warped Image w/ GT and Estim from Unsupervised Model', warped_image_unsup, 
                                          (DataGeneration.warp_bb(perturbed_patch, H_offset), 
                                           DataGeneration.warp_bb(unsup_warped_bb_estim, unsup_homography_estim_offset)))
    # cv2.imshow('after homography net', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
