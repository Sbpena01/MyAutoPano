import numpy as np
import pandas as pd
import csv
import cv2
import torch

def read_data(path, idx) -> tuple[np.ndarray, np.ndarray]:
    patches = []
    corners = []
    with open(f"{path}Patch_Stacks/patch_stack_{idx}.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        current = []
        for row in reader:
            if len(row) == 8:
                corners.append(row)
                continue
            if not row:
                current_np = np.array(current, dtype=np.float32)
                patch_a = current_np[0:128, :]
                patch_b = current_np[128:, :]
                actual_stack = np.array([patch_a, patch_b])
                patches.append(actual_stack)
                current = []
                continue
            current.append(row)
    homography = []
    with open(f"{path}Homographies/homography_{idx}.csv", 'r') as csvfile:
        reader = csv.reader(csvfile)
        current = []
        for row in reader:
            if not row:
                current_np = np.array(current, dtype=np.float32)
                current_np = np.reshape(current_np, (4, 2))
                homography.append(current_np)
                current = []
                continue
            current.append(row)
    patches = np.array(patches)
    homography = np.array(homography)
    # patches = np.transpose(patches, axes=(0, 3, 2, 1))
    corners_np = np.array(corners, dtype=np.float32)
    return patches, homography, torch.from_numpy(corners_np)

def calculate_dsize(image: np.ndarray, homography):
    corners = np.array([
                    [0, 0],
                    [0, image.shape[0]],
                    [image.shape[1], 0],
                    [image.shape[1], image.shape[0]]
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

    width = max(x_max, image.shape[1]) - min(0, x_min)
    height = max(y_max,image.shape[0]) - min(0, y_min)
    dsize = (width, height)

    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    offset_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return dsize, offset_matrix

def get_image_from_idx(image_idx, is_train=True):
    if is_train:
        return cv2.imread(f"Phase2/Data/Train/{image_idx}.jpg", cv2.IMREAD_GRAYSCALE)
    return cv2.imread(f"Phase2/Data/Val/{image_idx}.jpg", cv2.IMREAD_GRAYSCALE)

def tensor_dlt(homography_4pt: torch.tensor, corners_a: list[np.ndarray]):
    # TODO: step through
    if homography_4pt.shape != corners_a.shape:
        raise ValueError(f"4 Point Homography and Corner A matrices do not share the same shape. H_4pt: {homography_4pt.shape}  C_a: {corners_a.shape}")

    output = []
    for H, c_a in zip(homography_4pt, corners_a):
        c_b = c_a + H
        H_3x3 = compute_homography(c_a, c_b)
        # H_3x3 = find_homography(c_a, c_b)
        output.append(H_3x3)
    return torch.tensor(output)  # 64x3x3

def spacial_transform_layer(homographies:torch.tensor, image:np.ndarray, corners_list: torch.tensor):
    # Step One: calculated inverse homography
    W = image.shape[1]
    H = image.shape[0]
    M = np.array([
        [W/2, 0, W/2],
        [0, H/2, H/2],
        [0, 0, 1]
    ])
    # H^(-1) = M^(-1)*H^(-1)*M
    estim_patch_stack = torch.empty((1,1,128,128), requires_grad=True)
    
    for homography, corners in zip(homographies, corners_list):
        homography_inv = np.matmul(np.matmul(np.linalg.inv(M), np.linalg.inv(homography)), M)

        affine_grid = np.array([homography_inv[0:2, 0:3]])


        # dummy_grid = torch.zeros((1,1,image.shape[1], image.shape[0]))
        # input = np.array([[image]])

        input = torch.from_numpy(np.array([[image]], dtype=np.float64))
        
        # cv2.imshow('orig', np.uint8(np.squeeze(image)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        grid = torch.nn.functional.affine_grid(torch.from_numpy(affine_grid), torch.Size((1,1,image.shape[0], image.shape[1])))
        full_warped_image = torch.nn.functional.grid_sample(input, grid, padding_mode='reflection')
        # cv2.imshow('warp', np.uint8(np.squeeze(full_warped_image)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        corners = torch.reshape(corners, (4,2))
        # print(corners[0,0].item())
        # print(corners[1,0].item())
        
        # print(corners[0,1].item())
        # print(corners[3,1].item())
        patch_B = full_warped_image[:, :, int(corners[0,1].item()):int(corners[3,1].item()), int(corners[0,0].item()):int(corners[1,0].item())]
        # patch_B = full_warped_image[:, :, int(corners[0,0].item()):int(corners[1,0].item()), int(corners[0,1].item()):int(corners[3,1].item())]
        patch_B = patch_B.int()
        
        # cv2.imshow('patch', np.uint8(np.squeeze(patch_B)))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # Step Two: Parameterized Sampling Grid Generator (PSGG)
        # Creating a matrix of similar dimensions as the image. We have 2 channels to store x,y coords.
        # G = np.zeros((image.shape[0], image.shape[1], 2))
        # for row_idx in range(H-1):
        #     for col_idx in range(W-1):
        #         coord_np = np.array([row_idx, col_idx, 1]).transpose()
        #         resultant = np.matmul(homography_inv, coord_np)
                
        #         G[row_idx, col_idx, 0] = resultant[0]# X
        #         G[row_idx, col_idx, 1] = resultant[1]# Y

        # G = np.array([G], dtype=np.double)
        # grid = torch.zeros((1, 128, 128, 2), dtype=torch.double)
        # # Step Three: sample G to create estimation of patch B.
        # V = torch.nn.functional.grid_sample(torch.from_numpy(image), torch.from_numpy(G), padding_mode="reflection")
        # warped_patch = np.zeros((128,128))
        # for y in range(V.shape[0]):
        #     for x in range(V.shape[1]):
        #         transformed_coordinates = V[y,x]
        #         warped_patch[y,x] = image[transformed_coordinates]
        estim_patch_stack = torch.vstack((estim_patch_stack, patch_B))
    output = estim_patch_stack[1:65, :, :, :]
    # return output.flip(dims=(0,))
    return output

def compute_homography(points_1, points_2):
    points_1 = torch.reshape(points_1, (4,2))
    points_2 = torch.reshape(points_2, (4,2))
    p1 = points_1[0].detach()
    p2 = points_1[1].detach()
    p3 = points_1[2].detach()
    p4 = points_1[3].detach()
    p1_p = points_2[0].detach()
    p2_p = points_2[1].detach()
    p3_p = points_2[2].detach()
    p4_p = points_2[3].detach()
    # p1, p2, p3, p4 = points_1
    # p1_p, p2_p, p3_p, p4_p = points_2
    
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
    if np.linalg.det(P) == 0:  # Matrix is singular
        return None
    H = np.linalg.solve(P,b)
    H = np.reshape(H, (3,3))
    return H

def find_homography(points_source, points_target):
    A  = construct_A(points_source, points_target)
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    
    # Solution to H is the last column of V, or last row of V transpose
    homography = vh[-1].reshape((3,3))
    return homography/homography[2,2]

def construct_A(points_source, points_target):
    assert points_source.shape == points_target.shape, "Shape does not match"
    num_points = points_source.shape[0]

    matrices = []
    for i in range(num_points):
        partial_A = construct_A_partial(points_source[i], points_target[i])
        matrices.append(partial_A)
    return np.concatenate(matrices, axis=0)

def construct_A_partial(point_source, point_target):
    x, y, z = point_source[0], point_source[1], 1
    x_t, y_t, z_t = point_target[0], point_target[1], 1

    A_partial = np.array([
        [0, 0, 0, -z_t*x, -z_t*y, -z_t*z, y_t*x, y_t*y, y_t*z],
        [z_t*x, z_t*y, z_t*z, 0, 0, 0, -x_t*x, -x_t*y, -x_t*z]
    ])
    return A_partial
class Point:
    def __init__(self, point: tuple):
        self.x = int(point[1])
        self.y = int(point[0])

    # WITH format (X,Y) NOT (Y,X)
    def to_numpy(self):
        return np.array([self.x, self.y])

    def __eq__(self, other: "Point"):
        return (self.x == other.x) and (self.y == other.y)
    
    def to_xy_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)

class Bounding_Box:
    def __init__(self, tl: Point, tr:Point, bl:Point, br:Point):
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br

    def get_points(self) -> list[Point]:
        return [self.tl, self.tr, self.bl, self.br]
    
    def get_points_np(self) -> np.ndarray:
        return np.array([   [self.tl.x, self.tl.y], 
                            [self.tr.x, self.tr.y],
                            [self.bl.x, self.bl.y],
                            [self.br.x, self.br.y]], dtype='float32')

    def __str__(self):
        return f"tl: ({self.tl.x}, {self.tl.y}), tr: ({self.tr.x}, {self.tr.y}),\n\t bl: ({self.bl.x}, {self.bl.y}), br: ({self.br.x}, {self.br.y})   "
    
    def __eq__(self, other: "Bounding_Box"):
        for point_1, point_2 in zip(self.get_points(), other.get_points()):
            if not point_1 == point_2:
                return False
        return True 

    