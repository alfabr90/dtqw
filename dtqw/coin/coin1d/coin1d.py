from datetime import datetime

from pyspark import StorageLevel

from dtqw.coin.coin import Coin
from dtqw.math.operator import Operator
from dtqw.utils.utils import CoordinateDefault, CoordinateMultiplier, CoordinateMultiplicand
from dtqw.mesh.mesh import is_mesh

__all__ = ['Coin1D']


class Coin1D(Coin):
    """Top-level class for 1-dimensional Coins."""

    def __init__(self, spark_context):
        """
        Build a top-level 1-dimensional Coin object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.

        """
        super().__init__(spark_context)
        self._size = 2

    def is_1d(self):
        """
        Check if this is a Coin for 1-dimensional meshes.

        Returns
        -------
        bool

        """
        return True

    def is_2d(self):
        """
        Check if this is a Coin for 2-dimensional meshes.

        Returns
        -------
        bool

        """
        return False

    def create_operator(self, mesh, num_partitions,
                        coord_format=CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the coin operator for the walk.

        Parameters
        ----------
        mesh : Mesh
            A Mesh instance.
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Operator.CoordinateDefault.
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

        if not mesh.is_1d():
            if self._logger:
                self._logger.error("non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        mesh_size = mesh.size
        shape = (self._data.shape[0] * mesh_size, self._data.shape[1] * mesh_size)
        data = self._spark_context.broadcast(self._data)

        # The coin operator is built by applying a tensor product between the chosen coin and
        # an identity matrix with the dimensions of the chosen mesh.
        def __map(x):
            for i in range(data.value.shape[0]):
                for j in range(data.value.shape[1]):
                    yield (i * mesh_size + x, j * mesh_size + x, data.value[i][j])

        rdd = self._spark_context.range(
            mesh_size
        ).flatMap(
            __map
        )

        if coord_format == CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0], (m[1], m[2]))
            ).partitionBy(
                numPartitions=num_partitions
            )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        self._profile(operator, initial_time)

        return operator
