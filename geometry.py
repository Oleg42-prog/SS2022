import numpy as np

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1/2)

class Geometry:
    x_min = 0.0
    x_mean = 0.0
    x_max = 0.0
    y_min = 0.0
    y_mean = 0.0
    y_max = 0.0
        
    x_min_int = 0
    x_mean_int = 0
    x_max_int = 0
    y_min_int = 0
    y_mean_int = 0
    y_max_int = 0
        
    top_left_corner = (0.0, 0.0)
    top_right_corner = (0.0, 0.0)
    bottom_right_corner = (0.0, 0.0)
    bottom_left_corner = (0.0, 0.0)
        
    top_left_corner_int = (0, 0)
    top_right_corner_int = (0, 0)
    bottom_right_corner_int = (0, 0)
    bottom_left_corner_int = (0, 0)
        
    mean = (0.0, 0.0)
    mean_int = (0, 0)
    
    width = 0.0
    height = 0.0
    
    width_int = 0
    height_int = 0

    def __sub__(self, other):
        g = Geometry()
        g.x_min = abs(self.x_min - other.x_min)
        g.x_mean = abs(self.x_mean - other.x_mean)
        g.x_max = abs(self.x_max - other.x_max)
        g.y_min = abs(self.y_min - other.y_min)
        g.y_mean = abs(self.y_mean - other.y_mean)
        g.y_max = abs(self.y_max - other.y_max)

        g.x_min_int = abs(self.x_min_int - other.x_min_int)
        g.x_mean_int = abs(self.x_mean_int - other.x_mean_int)
        g.x_max_int = abs(self.x_max_int - other.x_max_int)
        g.y_min_int = abs(self.y_min_int - other.y_min_int)
        g.y_mean_int = abs(self.y_mean_int - other.y_mean_int)
        g.y_max_int = abs(self.y_max_int - other.y_max_int)

        g.top_left_corner = distance(self.top_left_corner, other.top_left_corner)
        g.top_right_corner = distance(self.top_right_corner, other.top_right_corner)
        g.bottom_right_corner = distance(self.bottom_right_corner, other.bottom_right_corner)
        g.bottom_left_corner = distance(self.bottom_left_corner, other.bottom_left_corner)
        g.top_left_corner_int = distance(self.top_left_corner_int, other.top_left_corner_int)
        g.top_right_corner_int = distance(self.top_right_corner_int, other.top_right_corner_int)
        g.bottom_right_corner_int = distance(self.bottom_right_corner_int, other.bottom_right_corner_int)
        g.bottom_left_corner_int = distance(self.bottom_left_corner_int, other.bottom_left_corner_int)
        g.mean = distance(self.mean, other.mean)
        g.mean_int = distance(self.mean_int, other.mean_int)
        return g

class MaskGeometry(Geometry):
    def __init__(self, mask):
        indices = np.indices((mask.shape[1], mask.shape[0])).transpose()
        selected_indices = indices[mask.sum(axis=2) != 0]
        
        self.x_min = selected_indices[:, 0].min()
        self.x_mean = selected_indices[:, 0].mean()
        self.x_max = selected_indices[:, 0].max()
        self.y_min = selected_indices[:, 1].min()
        self.y_mean = selected_indices[:, 1].mean()
        self.y_max = selected_indices[:, 1].max()
        
        self.x_min_int = int(self.x_min)
        self.x_mean_int = int(self.x_mean)
        self.x_max_int = int(self.x_max)
        self.y_min_int = int(self.y_min)
        self.y_mean_int = int(self.y_mean)
        self.y_max_int = int(self.y_max)
        
        self.top_left_corner = (self.x_min, self.y_min)
        self.top_right_corner = (self.x_max, self.y_min)
        self.bottom_right_corner = (self.x_max, self.y_max)
        self.bottom_left_corner = (self.x_min, self.y_max)
        
        self.top_left_corner_int = (self.x_min_int, self.y_min_int)
        self.top_right_corner_int = (self.x_max_int, self.y_min_int)
        self.bottom_right_corner_int = (self.x_max_int, self.y_max_int)
        self.bottom_left_corner_int = (self.x_min_int, self.y_max_int)
        
        self.mean = (self.x_mean, self.y_mean)
        self.mean_int = (self.x_mean_int, self.y_mean_int)
        
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        
        self.width_int = self.x_max_int - self.x_min_int
        self.height_int = self.y_max_int - self.y_min_int