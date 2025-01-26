import numpy as np
class Point:
    def __init__(self, point: tuple):
        self.x = int(point[1])
        self.y = int(point[0])

    # WITH format (X,Y) NOT (Y,X)
    def to_numpy(self):
        return np.array([self.x, self.y])


class Bounding_Box:
    def __init__(self, tl: Point, tr:Point, bl:Point, br:Point):
        self.tl = tl
        self.tr = tr
        self.bl = bl
        self.br = br


    