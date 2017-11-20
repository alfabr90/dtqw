from pyspark import StorageLevel
from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.linalg.operator import Operator

__all__ = ['Segment']


class Segment(Mesh1D):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self.__size = self._define_size(size)

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        coin_size = 2
        size = self._size
        shape = (coin_size * size, coin_size * size)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i

                    if x + l >= size or x + l < 0 or bl_broad.value.get(x + i + l) is not None:
                        bl = 0
                    else:
                        bl = l

                    yield ((i + bl) * size + x + bl, (1 - i) * size + x, 1)
        else:
            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i

                    if x + l >= size or x + l < 0:
                        bl = 0
                    else:
                        bl = l

                    yield ((i + bl) * size + x + bl, (1 - i) * size + x, 1)

        rdd = self._spark_context.range(
            size
        ).flatMap(
            __map
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        if self._broken_links_probability:
            bl_broad.unpersist()

        return operator
