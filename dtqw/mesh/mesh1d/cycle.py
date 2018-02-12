from datetime import datetime

from pyspark import StorageLevel

from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.math.operator import Operator
from dtqw.utils.utils import Utils

__all__ = ['Cycle']


class Cycle(Mesh1D):
    """Class for Cycle mesh."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a Cycle mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int
            Size of the mesh.
        broken_links : BrokenLinks, optional
            A BrokenLinks object.
        """
        super().__init__(spark_context, size, broken_links=broken_links)

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
                        coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the shift operator for the walk.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.
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
        num_edges = self._num_edges
        shape = (coin_size * size, coin_size * size)

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            def __map(e):
                """e = (edge, (edge, broken or not))"""
                for i in range(coin_size):
                    l = (-1) ** i

                    # Finding the correspondent x coordinate of the vertex from the edge number
                    x = (e[1][0] - i - l) % size

                    if e[1][1]:
                        l = 0

                    yield (i + l) * size + (x + l) % size, (1 - i) * size + x, 1

            rdd = self._spark_context.range(
                num_edges
            ).map(
                lambda m: (m, m)
            ).partitionBy(
                numPartitions=num_partitions
            ).leftOuterJoin(
                broken_links
            ).flatMap(
                __map
            )
        else:
            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i
                    yield i * size + (x + l) % size, i * size + x, 1

            rdd = self._spark_context.range(
                size
            ).flatMap(
                __map
            )

        if coord_format == Utils.CoordinateMultiplier or coord_format == Utils.CoordinateMultiplicand:
            rdd = Utils.changeCoordinate(
                rdd, Utils.CoordinateDefault, new_coord=coord_format
            ).partitionBy(
                numPartitions=num_partitions
            )

        operator = Operator(rdd, shape, coord_format=coord_format).materialize(storage_level)

        if self._broken_links:
            broken_links.unpersist()

        self._profile(operator, initial_time)

        return operator
