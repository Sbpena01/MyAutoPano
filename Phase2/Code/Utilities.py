import numpy as np
import pandas as pd


NUM_DATA = 100

def read_data(path, idx) -> tuple[np.ndarray, np.ndarray]:
    patches = pd.read_csv(f"{path}Patch_Stacks/patch_stack_{idx}.csv", header=None)
    patches = patches.to_numpy(dtype=np.float32)
    patches = np.reshape(patches, (128,128,6))
    patches = np.transpose(patches, axes=(2,0,1))
    homography = pd.read_csv(f"{path}Homographies/homography_{idx}.csv", header=None)
    homography = homography.to_numpy(dtype=np.float32)
    return patches, homography


# def compute_homography(points_1, points_2):
#     p1, p2, p3, p4 = points_1
#     p1_p, p2_p, p3_p, p4_p = points_2
    
#     # set up PH matrix
#     P = np.array([
#         [-p1[1], -p1[0], -1, 0, 0, 0, p1[1]*p1_p[1], p1[0]*p1_p[1], p1_p[1]],
#         [0, 0, 0, -p1[1], -p1[0], -1, p1[1]*p1_p[0], p1[0]*p1_p[0], p1_p[0]],
#         [-p2[1], -p2[0], -1, 0, 0, 0, p2[1]*p2_p[1], p2[0]*p2_p[1], p2_p[1]],
#         [0, 0, 0, -p2[1], -p2[0], -1, p2[1]*p2_p[0], p2[0]*p2_p[0], p2_p[0]],
#         [-p3[1], -p3[0], -1, 0, 0, 0, p3[1]*p3_p[1], p3[0]*p3_p[1], p3_p[1]],
#         [0, 0, 0, -p3[1], -p3[0], -1, p3[1]*p3_p[0], p3[0]*p3_p[0], p3_p[0]],
#         [-p4[1], -p4[0], -1, 0, 0, 0, p4[1]*p4_p[1], p4[0]*p4_p[1], p4_p[1]],
#         [0, 0, 0, -p4[1], -p4[0], -1, p4[1]*p4_p[0], p4[0]*p4_p[0], p4_p[0]],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1],
#     ])

#     b = np.zeros((9,1))
#     b[8,0] = 1
#     if np.linalg.det(P) == 0:  # Matrix is singular
#         return None
#     H = np.linalg.solve(P,b)
#     H = np.reshape(H, (3,3))
#     return H


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

    