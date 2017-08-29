from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.math.operator import Operator

__all__ = ['BoxDiagonal']


class BoxDiagonal(Mesh2D):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)
        self.__size = self._define_size(size)

    def title(self):
        return 'Diagonal Box'

    def create_operator(self):
        coin_size = 2
        size = self.__size
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        def __map(xy):
            x = int(xy / size[1])
            y = xy % size[1]

            for i in range(coin_size):
                l1 = (-1) ** i
                for j in range(coin_size):
                    l2 = (-1) ** j

                    if x + l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                        bl1 = 0
                        bl2 = 0
                    else:
                        bl1 = l1
                        bl2 = l2

                    m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + (x + bl1) * size[1] + (y + bl2)
                    n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                    yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape, self._logger.filename)
