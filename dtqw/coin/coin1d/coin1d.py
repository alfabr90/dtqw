from dtqw.coin.coin import Coin
from dtqw.mesh.mesh import is_mesh
from dtqw.math.operator import Operator

__all__ = ['Coin1D']


class Coin1D(Coin):
    def __init__(self, spark_context, log_filename='./log.txt'):
        super().__init__(spark_context, log_filename)

    def is_1d(self):
        return True

    def create_operator(self, mesh):
        if not is_mesh(mesh):
            self._logger.error("expected mesh, not {}".format(type(mesh)))
            raise TypeError("expected mesh, not {}".format(type(mesh)))

        if not mesh.is_1d():
            self._logger.error("non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        mesh_size = mesh.size
        shape = (self._data.shape[0] * mesh_size, self._data.shape[1] * mesh_size)
        data = self._spark_context.broadcast(self._data)

        def __map(x):
            for i in range(data.value.shape[0]):
                for j in range(data.value.shape[1]):
                    yield (i * mesh_size + x, j * mesh_size + x, data.value[i][j])

        rdd = self._spark_context.range(
            mesh_size
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape, self._logger.filename)
