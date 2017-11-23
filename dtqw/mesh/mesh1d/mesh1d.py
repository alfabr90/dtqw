import random
from dtqw.utils.utils import broadcast
from dtqw.mesh.mesh import Mesh

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)

    def _validate(self, size):
        if type(size) != int:
            return False
        elif size <= 0:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            if self.logger:
                self.logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def axis(self):
        return range(self._size)

    def is_1d(self):
        return True

    def filename(self):
        return "{}_{}_{}".format(self.to_string(), self._size, self._broken_links_probability)

    def broken_links(self):
        size = self._size
        bl_prop = self._broken_links_probability

        def __map(e):
            if e == 0 or e == size:
                return e, False

            random.seed()
            return e, random.random() < bl_prop

        rdd = self._spark_context.range(
            size + 1
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        ).collectAsMap()

        return broadcast(self._spark_context, rdd)
