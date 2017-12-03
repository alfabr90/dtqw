import numpy as np
from pyspark import StorageLevel
from dtqw.mesh.mesh import Mesh

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)

    def _validate(self, size):
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                return False
        else:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            if self.logger:
                self.logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def axis(self):
        return np.meshgrid(range(self._size[0]), range(self._size[1]))

    def is_1d(self):
        return False

    def is_2d(self):
        return True

    def check_steps(self, steps):
        raise NotImplementedError

    def create_operator(self, num_partitions, mul_format=True, storage_level=StorageLevel.MEMORY_AND_DISK):
        raise NotImplementedError

    def filename(self):
        return "{}_{}-{}_{}".format(self.to_string(), self._size[0], self._size[1], self._broken_links_probability)
