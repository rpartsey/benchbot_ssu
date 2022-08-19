import numpy as np


class Map:
    def __init__(self, scale, height=1, width=1, dtype=np.uint8):
        self.scale = scale
        self.dtype = dtype
        self.h = height
        self.w = width
        self.origin_x = self.w // 2
        self.origin_y = self.h // 2
        self.map = np.zeros((self.h, self.w), dtype=self.dtype)

    @staticmethod
    def remove_duplicates(xs, ys):
        points = np.concatenate(
            (
                xs.reshape(-1, 1), 
                ys.reshape(-1, 1)
            ),
            axis=1
        )
        points = np.unique(points, axis=0)
        
        xs = points[:, 0]
        ys = points[:, 1]
        
        return xs, ys
    
    def expand(self, pad_x, pad_y):
        self.map = np.pad(
            self.map, 
            ((pad_y, pad_y), (pad_x, pad_x)), 
            mode='constant',
            constant_values=0
        )
        self.h += 2 * pad_y 
        self.w += 2 * pad_x
        
        assert self.h == self.map.shape[0]
        assert self.w == self.map.shape[1]
        
        self.origin_x = self.w // 2
        self.origin_y = self.h // 2
    
    def update(self, points):
        xs = np.floor(points[:, 0] / self.scale).astype(int) + self.origin_x
        ys = np.floor(points[:, 1] / self.scale).astype(int) + self.origin_y
        
        xs, ys = self.remove_duplicates(xs, ys)
        
        pad_x = np.max(np.abs(xs) - (self.w - 1))
        pad_y = np.max(np.abs(ys) - (self.h - 1))
        
        pad_x = max(pad_x, 0)
        pad_y = max(pad_y, 0)
        
        if (pad_x > 0 and pad_y >= 0) or (pad_x >= 0 and pad_y > 0):
            self.expand(pad_x, pad_y)
            self.update(points)
            return 
        
        # flip upside down
        ys *= -1
        
        new_values = np.zeros_like(self.map)
        new_values[ys, xs] += 1
        
        mask = self.map < np.iinfo(self.dtype).max
        self.map[mask] += new_values[mask]
