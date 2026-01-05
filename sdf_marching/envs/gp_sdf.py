from erl_sdf_mapping import GpSdfMapping3D, GpSdfMapping2D
from typing import Union


class GaussianProcessSDF:
    def __init__(self, gp_sdf):
        self.gp_sdf: Union[GpSdfMapping2D, GpSdfMapping3D] = gp_sdf

    @staticmethod
    def from_bin_file(dim, bin_file):
        if dim == 2:
            setting = GpSdfMapping2D.Setting()
            setting.log_timing = False
            gp_sdf = GpSdfMapping2D(setting)
        elif dim == 3:
            setting = GpSdfMapping3D.Setting()
            setting.log_timing = False
            gp_sdf = GpSdfMapping3D(setting)
        else:
            raise ValueError(f"Unsupported dim: {dim}")

        gp_sdf.read(bin_file)
        setting.log_timing = False  # disable timing
        return GaussianProcessSDF(gp_sdf)

    def __call__(self, test_positions):
        return self.gp_sdf.test(test_positions.transpose())[0]

    def points(self):
        # FIXME: This is not defined
        return self._points

    @property
    def mins(self):
        if isinstance(self.gp_sdf, GpSdfMapping2D):
            mins, _ = self.gp_sdf.surface_mapping.quadtree.metric_min_max
        elif isinstance(self.gp_sdf, GpSdfMapping3D):
            mins, _ = self.gp_sdf.surface_mapping.octree.metric_min_max
        else:
            raise ValueError(f"Unsupported sdf mapping: {self.gp_sdf}")
        return mins

    @property
    def maxs(self):
        if isinstance(self.gp_sdf, GpSdfMapping2D):
            _, maxs = self.gp_sdf.surface_mapping.quadtree.metric_min_max
        elif isinstance(self.gp_sdf, GpSdfMapping3D):
            _, maxs = self.gp_sdf.surface_mapping.octree.metric_min_max
        else:
            raise ValueError(f"Unsupported sdf mapping: {self.gp_sdf}")
        return maxs

    @property
    def space_tree(self):
        if isinstance(self.gp_sdf, GpSdfMapping2D):
            return self.gp_sdf.surface_mapping.quadtree
        elif isinstance(self.gp_sdf, GpSdfMapping3D):
            return self.gp_sdf.surface_mapping.octree
        else:
            raise ValueError(f"Unsupported sdf mapping: {self.gp_sdf}")
