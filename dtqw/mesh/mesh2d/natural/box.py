from pyspark import StorageLevel
from dtqw.mesh.mesh2d.natural.natural import Natural
from dtqw.linalg.operator import Operator

__all__ = ['BoxNatural']


class BoxNatural(Natural):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self.__size = self._define_size(size)

    def title(self):
        return 'Natural Box'

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        coin_size = 2
        size = self.__size
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l = (-1) ** i
                    for j in range(coin_size):
                        delta = int(not (i ^ j))

                        e = delta * (((size[0] + 1) * size[1]) + x * (size[1] + 1) + y + (1 - i)) + \
                            (1 - delta) * (y * (size[0] + 1) + x + (1 - i))

                        pos1 = x + l * (1 - delta)
                        pos2 = y + l * delta

                        if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0 or e in bl_broad.value:
                            bl = 0
                        else:
                            bl = l

                        m = ((i + bl) * coin_size + (abs(j + bl) % coin_size)) * size_xy + \
                            (x + bl * (1 - delta)) * size[1] + (y + bl * delta)
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield (m, n, 1)
        else:
            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l = (-1) ** i
                    for j in range(coin_size):
                        delta = int(not (i ^ j))

                        pos1 = x + l * (1 - delta)
                        pos2 = y + l * delta

                        if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                            bl = 0
                        else:
                            bl = l

                        m = ((i + bl) * coin_size + (abs(j + bl) % coin_size)) * size_xy + \
                            (x + bl * (1 - delta)) * size[1] + (y + bl * delta)
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
