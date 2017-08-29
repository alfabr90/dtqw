from dtqw.mesh.mesh import Mesh

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)

    def _validate(self, size):
        if type(size) != int:
            return False
        elif size <= 0:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def axis(self):
        return range(self._size)

    def is_1d(self):
        return True

    def filename(self):
        return "{}_{}".format(self.to_string(), self._size)
