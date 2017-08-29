import numpy as np
from dtqw.mesh.mesh import Mesh

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)

    def _validate(self, size):
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                return False
        else:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def axis(self):
        return np.meshgrid(range(self._size[0]), range(self._size[1]))

    def is_2d(self):
        return True

    def filename(self):
        return "{}_{}-{}".format(self.to_string(), self._size[0], self._size[1])
