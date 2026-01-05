import numpy as np 
from kdtree_mapping import KDTreeMapping
from plyfile import PlyData

class KDTreeSDF:
    def __init__(self, points, cell_size=0.1, num_neighbors=1):
        self.kdtree_map = KDTreeMapping(cell_size, num_neighbors)
        self.kdtree_map.add_point_cloud(np.eye(4), points.transpose())
        self._points = points

    @staticmethod
    def from_ply(path, **kwargs):
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)

        points = np.stack(
            [
                plydata['vertex']['x'],
                plydata['vertex']['y'],
                plydata['vertex']['z']
            ],
            axis=1
        )

        return KDTreeSDF(points, **kwargs)

    def __call__(self, test_positions):
        return self.kdtree_map.query_sdf(test_positions.transpose())

    def points(self):
        return self._points

    @property
    def mins(self):
        mins, _ = self.kdtree_map.get_bounds()
        return mins

    @property
    def maxs(self):
        _, maxs = self.kdtree_map.get_bounds()
        return maxs
