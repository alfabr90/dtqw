from pyspark import StorageLevel
from dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from dtqw.linalg.operator import Operator

__all__ = ['BoxDiagonal']


class BoxDiagonal(Diagonal):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self.__size = self._define_size(size)

    def title(self):
        return 'Diagonal Box'

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        coin_size = 2
        size = self._size
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        e = (y + 1 - j) * (size[0] + 1) + x + 1 - i

                        if x + l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0 or e in bl_broad.value:
                            bl1, bl2 = 0, 0
                        else:
                            bl1, bl2 = l1, l2

                        m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + (x + bl1) * size[1] + (y + bl2)
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield (m, n, 1)
        else:
            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        if x + l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                            bl1, bl2 = 0, 0
                        else:
                            bl1, bl2 = l1, l2

                        m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + (x + bl1) * size[1] + (y + bl2)
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        if self._broken_links_probability:
            bl_broad.unpersist()

        return operator
