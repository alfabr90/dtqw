from datetime import datetime
from pyspark import StorageLevel
from dtqw.coin.coin import Coin
from dtqw.mesh.mesh import is_mesh
from dtqw.linalg.operator import Operator

__all__ = ['Coin2D']


class Coin2D(Coin):
    def __init__(self, spark_context):
        super().__init__(spark_context)
        self._size = 4

    def is_1d(self):
        return False

    def is_2d(self):
        return True

    def create_operator(self, mesh, num_partitions, mul_format=True, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the coin operator for the walk.

        Parameters
        ----------
        mesh : Mesh
            A Mesh instance.
        num_partitions : int
            The desired number of partitions for the RDD.
        mul_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is True.

            If mul_format is True, the returned operator will not be in (i,j,value) format, but in (j,(i,value)) format.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        Operator

        """
        if self._logger:
            self._logger.info("building coin operator...")

        initial_time = datetime.now()

        if not is_mesh(mesh):
            if self._logger:
                self._logger.error("expected mesh, not {}".format(type(mesh)))
            raise TypeError("expected mesh, not {}".format(type(mesh)))

        if not mesh.is_2d():
            if self._logger:
                self._logger.error("non correspondent coin and mesh dimensions")
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

        if mul_format:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            )
        else:
            rdd = rdd.map(
                lambda m: (m[0], (m[1], m[2]))
            )

        rdd = rdd.partitionBy(
            numPartitions=num_partitions
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        self._profile(operator, initial_time)

        return operator
