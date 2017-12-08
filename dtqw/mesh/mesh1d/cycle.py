from datetime import datetime
from pyspark import StorageLevel
from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.linalg.matrix import Matrix
from dtqw.linalg.operator import Operator

__all__ = ['Cycle']


class Cycle(Mesh1D):
    """Class for Cycle mesh."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a Cycle mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int
            Size of the mesh.
        bl_prob : float, optional
            Probability of the occurences of broken links in the mesh.
        """
        super().__init__(spark_context, size, bl_prob)
        self._size = self._define_size(size)

    def check_steps(self, steps):
        """
        Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Returns
        -------
        bool

        """
        return True

    def create_operator(self, num_partitions,
                        coord_format=Matrix.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the shift operator for the walk.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Matrix.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        Operator

        """
        if self._logger:
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        coin_size = 2
        size = self._size
        shape = (coin_size * size, coin_size * size)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i

                    # Finding the correspondent edge number from the x coordinate of the vertex
                    e = (x + i + l) % size

                    if e in bl_broad.value:
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

        if coord_format == Matrix.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            )
        elif coord_format == Matrix.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0], (m[1], m[2]))
            )

        rdd = rdd.partitionBy(
            numPartitions=num_partitions
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        if self._broken_links_probability:
            bl_broad.unpersist()

        self._profile(operator, initial_time)

        return operator
