from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.linalg.operator import Operator

__all__ = ['BoxNatural']


class BoxNatural(Mesh2D):
    def __init__(self, spark_context, size):
        super().__init__(spark_context, size)
        self.__size = self._define_size(size)

    def title(self):
        return 'Natural Box'

    def create_operator(self):
        coin_size = 2
        size = self.__size
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        def __map(xy):
            x = int(xy / size[1])
            y = xy % size[1]

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

                    m = ((i + bl) * coin_size + (abs(j + bl) % coin_size)) * size_xy + (x + bl * (1 - delta)) * size[1] + (y + bl * delta) % size[1]
                    n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                    yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape)
