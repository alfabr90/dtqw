import numpy as np
from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.math.operator import Operator

__all__ = ['LatticeDiagonal']


class LatticeDiagonal(Mesh2D):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)
        self.__size = self._define_size(size)

    def _define_size(self, size):
        if not self._validate(size):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return 2 * size[0] + 1, 2 * size[0] + 1

    def title(self):
        return 'Diagonal Lattice'

    def axis(self):
        return np.meshgrid(
            range(- int((self._size[0] - 1) / 2), int((self._size[0] - 1) / 2) + 1),
            range(- int((self._size[1] - 1) / 2), int((self._size[1] - 1) / 2) + 1)
        )

    def check_steps(self, steps):
        return steps <= int((self.__size[0] - 1) / 2) and steps <= int((self.__size[1] - 1) / 2)

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

                    m = (i * coin_size + j) * size_xy + ((x + l1) % size[0]) * size[1] + (y + l2)
                    n = (i * coin_size + j) * size_xy + x * size[1] + y

                    yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape, self._logger.filename)