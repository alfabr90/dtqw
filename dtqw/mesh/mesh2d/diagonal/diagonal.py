import random
from dtqw.utils.utils import broadcast
from dtqw.mesh.mesh2d.mesh2d import Mesh2D

__all__ = ['Diagonal']


class Diagonal(Mesh2D):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)

    def broken_links(self):
        size = self._size
        size_bl = (size[0] + 1) * (size[1] + 1)

        bl_prop = self._broken_links_probability

        def __map(e):
            x = e % (size[0] + 1)
            y = int(e / (size[0] + 1))

            if x == 0 or x == size[0] or y == 0 or y == size[1]:
                return e, False

            random.seed()
            return e, random.random() < bl_prop

        rdd = self._spark_context.range(
            size_bl
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        ).collectAsMap()

        return broadcast(self._spark_context, rdd)
