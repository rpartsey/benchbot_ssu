import numpy as np


class Map:
    def __init__(self, scale, dtype=np.uint8):
        self.scale = scale
        self.dtype = dtype
        self.map_x_min = 0
        self.map_x_max = 0
        self.map_y_min = 0
        self.map_y_max = 0
        self.map = np.zeros((0, 0), dtype=self.dtype)

    @staticmethod
    def _remove_duplicates(xs, ys):
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

    def _init(self, xs, ys):
        self.map_x_min = np.min(xs)
        self.map_x_max = np.max(xs)
        self.map_y_min = np.min(ys)
        self.map_y_max = np.max(ys)

        self.map = np.zeros(
            (self.map_y_max - self.map_y_min + 1, self.map_x_max - self.map_x_min + 1),
            dtype=self.dtype
        )
    
    def update(self, points):
        xs = np.floor(points[:, 0] / self.scale).astype(int)
        ys = np.floor(points[:, 1] / self.scale).astype(int)
        
        xs, ys = self._remove_duplicates(xs, ys)

        if self.map.shape == (0, 0):
            self._init(xs, ys)

        map_x_min = np.min(xs)
        map_x_max = np.max(xs)
        map_y_min = np.min(ys)
        map_y_max = np.max(ys)

        pad_x_neg = abs(min(0, map_x_min - self.map_x_min))
        pad_x_pos = max(0, map_x_max - self.map_x_max)
        pad_y_neg = abs(min(0, map_y_min - self.map_y_min))
        pad_y_pos = max(0, map_y_max - self.map_y_max)

        if not (pad_y_pos, pad_y_neg, pad_x_pos, pad_x_neg) == (0, 0, 0, 0):
            self.map = np.pad(
                self.map,
                ((pad_y_neg, pad_y_pos), (pad_x_neg, pad_x_pos)),
                mode='constant',
                constant_values=0
            )
            self.map_x_min -= pad_x_neg
            self.map_x_max += pad_x_pos
            self.map_y_min -= pad_y_neg
            self.map_y_max += pad_y_pos

        xs -= self.map_x_min
        ys -= self.map_y_min

        new_values = np.zeros_like(self.map)
        new_values[ys, xs] += 1
        
        mask = self.map < np.iinfo(self.dtype).max
        self.map[mask] += new_values[mask]
