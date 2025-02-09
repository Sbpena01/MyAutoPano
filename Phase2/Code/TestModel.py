import torch
import cv2
import os
import random
import Utilities
from Network.Network import HomographyModel
import DataGeneration
import numpy as np


def main():
    # load model 
    device = torch.device('cuda')
    device_cpu = torch.device('cpu')
    model = HomographyModel('Sup')
    # model.to(device)
    model.eval()

    # Checkpoint = torch.load("Phase2/Checkpoints/49modelcpu.ckpt", map_location=device_cpu)
    # model.load_state_dict(Checkpoint["model_state_dict"])
    model.load_state_dict(torch.load("Phase2/Checkpoints/49modelcpu.ckpt", weights_only=True), strict=False)
    
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

    patches_model = torch.from_numpy(np.array([[unwarped_patch, warped_patch]], dtype=np.float32))
    print(patches_model.shape)
    # patches_model = patches_model.to(device)
    with torch.no_grad():
        output = model(patches_model)
    
    # gives us a h4pt...
    # translate this into a 3x3 homography...
    H_4pt = perturbed_patch.get_points_np() - original_patch.get_points_np()
    print(H_4pt)
    print(output)

    output = output.numpy().reshape((4,2))

    
    homography_estim = Utilities.find_homography(original_patch.get_points_np(), original_patch.get_points_np()+output)
    homography_estim_inv = np.linalg.inv(homography_estim)
    dsize_estim, offset_matrix_estim = Utilities.calculate_dsize(image, homography_estim_inv)
    homography_estim_offset = np.dot(offset_matrix_estim, homography_estim_inv)
    warped_image = cv2.warpPerspective(image, homography_estim_offset, dsize=dsize_estim)
    print(homography)
    print("\n")
    print(homography_estim)
 
    warped_bb_estim = DataGeneration.warp_bb(original_patch, homography_estim)

    # H_output_inv = np.linalg.inv(H_output)
    # # H_output_forward = np.dot(offset_matrix_output, H_output)
    # H_output_inv = np.dot(offset_matrix_output, H_output_inv)
    DataGeneration.display_bounding_boxes('unwarped image w/ GT and Estim', image, (perturbed_patch, warped_bb_estim))
    
    DataGeneration.display_bounding_boxes(' Warped Image w/ GT and Estim', warped_image, 
                                          (DataGeneration.warp_bb(perturbed_patch, H_offset), 
                                           DataGeneration.warp_bb(warped_bb_estim, homography_estim_offset)))
    # cv2.imshow('after homography net', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return

if __name__ == '__main__':
    main()
