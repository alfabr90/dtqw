from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.linalg.operator import Operator

__all__ = ['Segment']


class Segment(Mesh1D):
    def __init__(self, spark_context, size):
        super().__init__(spark_context, size)
        self.__size = self._define_size(size)

    def create_operator(self):
        coin_size = 2
        size = self.__size
        shape = (coin_size * size, coin_size * size)

        def __map(x):
            for i in range(coin_size):
                l = (-1) ** i

                if x + l >= size or x + l < 0:
                    bl = 0
                else:
                    bl = l

                yield ((i + bl) * size + (x + bl) % size, (1 - i) * size + x, 1)

        rdd = self._spark_context.range(
            size
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape)
