import argparse
import cv2
import os
import numpy as np
import random
from Utilities import Point, Bounding_Box 
import copy

DEBUG_LEVEL = 0
PAIR_COUNT = 0
PERTURBATION_RANGE = 10
def load_images(im_path: str, num_images: int, flags: int = cv2.IMREAD_GRAYSCALE) -> tuple[list[cv2.Mat], list[str]]:
    images = []
    image_names = []
    filenames = os.listdir(im_path)
    filenames.sort(key=lambda x: int(x.split('.')[0]))  # Sort based on the number before the '.'
    count = 0
    for filename in filenames:
        full_image_path = im_path+filename
        image = cv2.imread(full_image_path, flags=flags)
        image_names.append(filename)
        images.append(image)
        count +=1
        if count == num_images:
            break
    return images, image_names

def generate_patch(image: np.ndarray, patch_size = 101):
    # We generate a random patch by randomly selecting a point to act at the top left corner of
    # patch. We first calculate the active region based on the perturbation range and the
    # patch size.
    image_width = image.shape[1]
    image_height = image.shape[0]
    #                         Lower bound         Upper bound
    random_x = random.randint(PERTURBATION_RANGE, image_width - (PERTURBATION_RANGE + patch_size))
    random_y = random.randint(PERTURBATION_RANGE, image_height - (PERTURBATION_RANGE + patch_size))
    
    # Build the Bounding_Box object to represent the generated patch.
    top_left = Point((random_y, random_x))
    top_right = Point((random_y, random_x+patch_size))
    bottom_left = Point((random_y+patch_size, random_x))
    bottom_right = Point((random_y+patch_size, random_x+patch_size))
    return Bounding_Box(top_left, top_right, bottom_left, bottom_right)


def perterbate_patch(bounding_box: Bounding_Box) -> Bounding_Box:
    new_bb_list = []
    for point in bounding_box.get_points():
        new_bb_list.append( Point((  (point.y) + random.randint(-PERTURBATION_RANGE, PERTURBATION_RANGE),
                                    (point.x) + random.randint(-PERTURBATION_RANGE, PERTURBATION_RANGE))))
    new_bb = Bounding_Box(new_bb_list[0],new_bb_list[1],new_bb_list[2],new_bb_list[3])

    if new_bb == bounding_box:
        new_bb = perterbate_patch(bounding_box)
    return new_bb

def warp_bb(bounding_box: Bounding_Box, homography):
    warp_array = np.array([
        [bounding_box.tl.x, bounding_box.tl.y, 1],
        [bounding_box.tr.x, bounding_box.tr.y, 1],
        [bounding_box.bl.x, bounding_box.bl.y, 1],
        [bounding_box.br.x, bounding_box.br.y, 1]
    ])

    warp_array = np.dot(homography, warp_array.T)

    warp_array /= warp_array[2]


    return Bounding_Box(Point((warp_array[1][0], warp_array[0][0])), Point((warp_array[1][1], warp_array[0][1])),
                        Point((warp_array[1][2], warp_array[0][2])), Point((warp_array[1][3], warp_array[0][3])))

def display_bounding_boxes(name:str, image:np.ndarray, bounding_boxes: list[Bounding_Box]):
    # tl to tr, tr to br, br to bl, bl to tl

    imcopy = copy.deepcopy(image)
    cv2.line(imcopy, bounding_boxes[0].tl.to_xy_tuple(), bounding_boxes[0].tr.to_xy_tuple(), (0, 0, 255), 2)
    cv2.line(imcopy, bounding_boxes[0].tr.to_xy_tuple(), bounding_boxes[0].br.to_xy_tuple(), (0, 0, 255), 2)
    cv2.line(imcopy, bounding_boxes[0].br.to_xy_tuple(), bounding_boxes[0].bl.to_xy_tuple(), (0, 0, 255), 2)
    cv2.line(imcopy, bounding_boxes[0].bl.to_xy_tuple(), bounding_boxes[0].tl.to_xy_tuple(), (0, 0, 255), 2)

    cv2.line(imcopy, bounding_boxes[1].tl.to_xy_tuple(), bounding_boxes[1].tr.to_xy_tuple(), (255, 0, 0), 2)
    cv2.line(imcopy, bounding_boxes[1].tr.to_xy_tuple(), bounding_boxes[1].br.to_xy_tuple(), (255, 0, 0), 2)
    cv2.line(imcopy, bounding_boxes[1].br.to_xy_tuple(), bounding_boxes[1].bl.to_xy_tuple(), (255, 0, 0), 2)
    cv2.line(imcopy, bounding_boxes[1].bl.to_xy_tuple(), bounding_boxes[1].tl.to_xy_tuple(), (255, 0, 0), 2)
    cv2.imshow(name, imcopy)


def main():
    Parser = argparse.ArgumentParser()
    """
    Image path, number of pairs / dataset size, Output Path, Debug Level
    """
    Parser.add_argument(
        "--ImagePath",
        default="Phase2/Data/Train/",
        help="Base path of images, Default: Phase2/Data/Train",
    )
    Parser.add_argument(
        "--OutputPath",
        default="/Outputs/Data/", 
        help="Base path of images, Default: /Outputs/Data",
    )

    Parser.add_argument(
        "--DebugLevel",
        type=int,
        default=0,
        help="Increase debug verbosity with higher debug level"
    )
    
    Parser.add_argument(
        "--PairCount",
        type=int,
        default=5,
        help="Increase debug verbosity with higher debug level"
    )
    
    Parser.add_argument(
        "--NumImages",
        type=int,
        default=20,
        help="Increase debug verbosity with higher debug level"
    )

    Args = Parser.parse_args()
    ImagePath = Args.ImagePath
    OutputPath = Args.OutputPath
    NumImages = Args.NumImages
    global DEBUG_LEVEL, PAIR_COUNT
    DEBUG_LEVEL = Args.DebugLevel
    PAIR_COUNT = Args.PairCount
    

    """ Read a set of images from input directory """
    image_set, image_names = load_images(ImagePath, NumImages, cv2.IMREAD_ANYCOLOR)

    """ Generate a sub patch within bounds [(x_min,y_min), (x_man, y_max)] (P_a) """
    
    test_image = image_set[0]
    test_name = image_names[0]


    unwarped_bb = generate_patch(test_image)
    unwarped_patch = test_image[unwarped_bb.tl.y:unwarped_bb.bl.y, unwarped_bb.tl.x:unwarped_bb.br.x]
    print(f"unwarped bb {unwarped_bb}\n")
    # cv2.imshow('original image', test_image)
    # cv2.imshow('unwarped_patch', unwarped_patch)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """ Randomly translate each corner on sub patch (P_b)"""
    warped_bb = perterbate_patch(unwarped_bb)
    print(f"warped bb {warped_bb}")

    display_bounding_boxes('unwarped_annotated', test_image, [unwarped_bb, warped_bb])
     
    homography = cv2.getPerspectiveTransform(unwarped_bb.get_points_np(), warped_bb.get_points_np())
    homography_inv = np.linalg.inv(homography)


    print(homography_inv)
    corners = np.array([
                [0, 0],
                [0, test_image.shape[0]],
                [test_image.shape[1], 0],
                [test_image.shape[1], test_image.shape[0]]
            ], dtype=np.float32)

    # Convert corners to homogeneous coordinates
    corners = np.column_stack((corners, np.ones(corners.shape[0])))

    # Apply the homography matrix
    transformed_corners = np.dot(homography_inv, corners.T) 

    # Normalize the points to convert back from homogeneous coordinates
    transformed_corners /= transformed_corners[2]

    # Extract x and y coordinates
    x_coords = transformed_corners[0]
    y_coords = transformed_corners[1]

    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

    width = max(x_max, test_image.shape[1]) - min(0, x_min)
    height = max(y_max,test_image.shape[0]) - min(0, y_min)
    dsize = (width, height)

    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    offset_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # Update the homography
    H_offset = np.dot(offset_matrix, homography_inv)

    warped_image = cv2.warpPerspective(test_image, M=H_offset, dsize=dsize)
    warped_patch = warped_image[(unwarped_bb.tl.y+offset_y):(unwarped_bb.bl.y+offset_y),
                                 (unwarped_bb.tl.x+offset_x):(unwarped_bb.br.x+offset_x)]

    # cv2.imshow('warped image', warped_image)
    cv2.imshow('warped patch', warped_patch)
    cv2.imshow('unwarped patch', unwarped_patch)
    display_bounding_boxes('warped_annotated', warped_image, [warp_bb(unwarped_bb, H_offset), warp_bb(warped_bb, H_offset)])

    cv2.waitKey(0)
    cv2.destroyAllWindows()



    """ Use Inverse of H (H^b_a) to transform image and generate warped Subpatch """

    """" Stack image frames (data_out) to a file, and generate corresponding label H_4pt """

    return


if __name__ == '__main__':
    main()
