from dtqw.coin.coin import Coin
from dtqw.mesh.mesh import is_mesh
from dtqw.linalg.operator import Operator

__all__ = ['Coin2D']


class Coin2D(Coin):
    def __init__(self, spark_context):
        super().__init__(spark_context)
        self._size = 4

    def is_2d(self):
        return True

    def create_operator(self, mesh):
        if not is_mesh(mesh):
            if self.logger:
                self.logger.error("expected mesh, not {}".format(type(mesh)))
            raise TypeError("expected mesh, not {}".format(type(mesh)))

        if not mesh.is_2d():
            if self.logger:
                self.logger.error("non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        mesh_size = mesh.size[0] * mesh.size[1]
        shape = (self._data.shape[0] * mesh_size, self._data.shape[1] * mesh_size)
        data = self._spark_context.broadcast(self._data)

        def __map(xy):
            for i in range(data.value.shape[0]):
                for j in range(data.value.shape[1]):
                    yield (i * mesh_size + xy, j * mesh_size + xy, data.value[i][j])

        rdd = self._spark_context.range(
            mesh_size
        ).flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape)
