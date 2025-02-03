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
    model = HomographyModel()

    Checkpoint = torch.load("Phase2/Checkpoints/Checkpoints1model.ckpt", map_location=torch.device("cpu"))

    model.load_state_dict(Checkpoint["model_state_dict"])
    
    idx = 1
    image = cv2.imread(f"/home/riley/Desktop/computer_vision/MyAutoPano/Phase2/Data/Val/{idx}.jpg", cv2.IMREAD_COLOR_RGB)
    original_patch = DataGeneration.generate_patch(image)
    perturbed_patch = DataGeneration.perterbate_patch(original_patch)
    homography = cv2.getPerspectiveTransform(original_patch.get_points_np(), perturbed_patch.get_points_np())
    homography_inv = np.linalg.inv(homography)
    dsize, offset_matrix = Utilities.calculate_dsize(image, homography_inv)
    # corners = np.array([
    #                 [0, 0],
    #                 [0, image.shape[0]],
    #                 [image.shape[1], 0],
    #                 [image.shape[1], image.shape[0]]
    #             ], dtype=np.float32)

    # # Convert corners to homogeneous coordinates
    # corners = np.column_stack((corners, np.ones(corners.shape[0])))

    # # Apply the homography matrix
    # transformed_corners = np.dot(homography_inv, corners.T) 

    # # Normalize the points to convert back from homogeneous coordinates
    # transformed_corners /= transformed_corners[2]

    # # Extract x and y coordinates
    # x_coords = transformed_corners[0]
    # y_coords = transformed_corners[1]

    # x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    # y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    # width = max(x_max, image.shape[1]) - min(0, x_min)
    # height = max(y_max,image.shape[0]) - min(0, y_min)
    # dsize = (width, height)

    # offset_x = -x_min if x_min < 0 else 0
    # offset_y = -y_min if y_min < 0 else 0

    # offset_matrix = np.array([
    #     [1, 0, offset_x],
    #     [0, 1, offset_y],
    #     [0, 0, 1]
    # ], dtype=np.float64)

    # Update the homography
    H_offset = np.dot(offset_matrix, homography_inv)

    warped_image = cv2.warpPerspective(image, M=H_offset, dsize=dsize)
    warped_patch = warped_image[int(original_patch.tl.y+offset_matrix[1,2]):int(original_patch.bl.y+offset_matrix[1,2]),
                                        int(original_patch.tl.x+offset_matrix[0, 2]):int(original_patch.br.x+offset_matrix[0,2])]

    DataGeneration.display_bounding_boxes('unwarped image w/ ground truth', image, (original_patch, perturbed_patch))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    unwarped_patch = image[original_patch.tl.y:original_patch.bl.y, original_patch.tl.x:original_patch.br.x]
    patch_stack = np.concatenate((unwarped_patch, warped_patch), axis=2, dtype=np.float32)

    patches_model = torch.from_numpy(np.array([np.transpose(patch_stack, (2,0,1))]))
    with torch.no_grad():
        output = model(patches_model)
    
    # gives us a h4pt...
    # translate this into a 3x3 homography...
    H_4pt = perturbed_patch.get_points_np() - original_patch.get_points_np()
    print(H_4pt)
    print(output)

    output = output.numpy()

    src = np.float32([[0, 0], [0, 128], [128, 0], [128, 128]])

    # Destination points
    dst = output.reshape((4,2))
    dst = src + dst

    # print(dst)

    # H_output = cv2.getPerspectiveTransform(src, dst)
    H_output = Utilities.find_homography(src, dst)

    print(homography)
    print(H_output)
    exit(1)


    warped_output_bb = DataGeneration.warp_bb(original_patch, H_output)

    # print(warped_output_bb)
    dsize_output, offset_matrix_output = Utilities.calculate_dsize(image, H_output)

    H_output_2 = np.dot(offset_matrix_output, H_output)
    # DataGeneration.display_bounding_boxes('unwarped image w/ ground truth', image, (perturbed_patch, warped_output_bb))
    warped_image = cv2.warpPerspective(image, H_output_2, dsize=dsize_output)
    DataGeneration.display_bounding_boxes(' Warped Image w/ ground truth', image, 
                                          (DataGeneration.warp_bb(perturbed_patch, H_offset), 
                                           DataGeneration.warp_bb(warped_output_bb, H_output_2)))
    cv2.imshow('after homography net', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return

if __name__ == '__main__':
    main()