import numpy as np
import pandas as pd

def read_data(path, idx) -> tuple[np.ndarray, np.ndarray]:
    patches = pd.read_csv(f"{path}Patch_Stacks/patch_stack_{idx}.csv", header=None)
    patches = patches.to_numpy(dtype=np.float32)
    patches = np.reshape(patches, (128,128,6))
    patches = np.transpose(patches, axes=(2,0,1))
    homography = pd.read_csv(f"{path}Homographies/homography_{idx}.csv", header=None)
    homography = homography.to_numpy(dtype=np.float32)
    return patches, homography

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

    