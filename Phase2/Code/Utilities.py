import numpy as np
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

    