from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.math.operator import Operator

__all__ = ['TorusNatural']


class TorusNatural(Mesh2D):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)
        self.__size = self._define_size(size)

    def title(self):
        return 'Natural Torus'

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

                    m = (i * coin_size + j) * size_xy + ((x + l * (1 - delta)) % size[0]) * size[1] + (y + l * delta) % size[1]
                    n = (i * coin_size + j) * size_xy + x * size[1] + y

                    yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape, self._logger.filename)
