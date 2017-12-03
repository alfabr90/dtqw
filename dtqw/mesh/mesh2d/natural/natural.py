import random
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.mesh.mesh2d.mesh2d import Mesh2D

__all__ = ['Natural']


class Natural(Mesh2D):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)

    def check_steps(self, steps):
        raise NotImplementedError

    def create_operator(self, num_partitions, mul_format=True, storage_level=StorageLevel.MEMORY_AND_DISK):
        raise NotImplementedError

    def broken_links(self):
        size = self._size
        size_bl = (size[0] + 1) * size[1] + size[0] * (size[1] + 1)

        bl_prop = self._broken_links_probability

        def __map(e):
            if e >= (size[0] + 1) * size[1]:
                x = int((e - (size[0] + 1) * size[1]) / (size[1] + 1))
                y = (e - (size[0] + 1) * size[1]) % (size[1] + 1)
            else:
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
