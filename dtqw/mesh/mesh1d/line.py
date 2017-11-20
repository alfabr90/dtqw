from pyspark import StorageLevel
from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.linalg.operator import Operator

__all__ = ['Line']


class Line(Mesh1D):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self._size = self._define_size(size)

    def _define_size(self, size):
        if not self._validate(size):
            if self.logger:
                self.logger.error("invalid size")
            raise ValueError("invalid size")

        return 2 * size + 1

    def axis(self):
        return range(- int((self._size - 1) / 2), int((self._size - 1) / 2) + 1)

    def check_steps(self, steps):
        return steps <= int((self._size - 1) / 2)

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        coin_size = 2
        size = self._size
        shape = (coin_size * size, coin_size * size)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i

                    if bl_broad.value.get(x + i + l) is not None:
                        l = 0

                    yield ((i + l) * size + (x + l) % size, (1 - i) * size + x, 1)
        else:
            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i
                    yield (i * size + (x + l) % size, i * size + x, 1)

        rdd = self._spark_context.range(
            size
        ).flatMap(
            __map
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        if self._broken_links_probability:
            bl_broad.unpersist()

        return operator
