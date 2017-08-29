from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.math.operator import Operator

__all__ = ['Line']


class Line(Mesh1D):
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        super().__init__(spark_context, size, log_filename)
        self._size = self._define_size(size)

    def _define_size(self, size):
        if not self._validate(size):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return 2 * size + 1

    def axis(self):
        return range(- int((self._size - 1) / 2), int((self._size - 1) / 2) + 1)

    def check_steps(self, steps):
        return steps <= int((self._size - 1) / 2)

    def create_operator(self, storage_level):
        coin_size = 2
        size = self._size
        shape = (coin_size * size, coin_size * size)

        def __map(x):
            for i in range(coin_size):
                l = (-1) ** i
                yield (i * size + (x + l) % size, i * size + x, 1)

        rdd = self._spark_context.range(
            size
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape, self._logger.filename).materialize(storage_level)
