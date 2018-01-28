from datetime import datetime

from pyspark import StorageLevel

from dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from dtqw.math.operator import Operator
from dtqw.utils.utils import CoordinateDefault, CoordinateMultiplier, CoordinateMultiplicand

__all__ = ['BoxDiagonal']


class BoxDiagonal(Diagonal):
    """Class for Diagonal Box mesh."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a Diagonal Box mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : tuple
            Size of the mesh.
        bl_prob : float, optional
            Probability of the occurences of broken links in the mesh.
        """
        super().__init__(spark_context, size, bl_prob=bl_prob)

        self._size = self._define_size(size)

    def title(self):
        return 'Diagonal Box'

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
                        coord_format=CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the shift operator for the walk.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is utils.CoordinateDefault.
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
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links_probability:
            broken_links = self.generate_broken_links()

            def __map(e):
                """e = (edge, (edge, broken or not))"""
                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        # Finding the correspondent x,y coordinates of the vertex from the edge number
                        x = (e[1][0] % size[0] - i - l1) % size[0]
                        y = (int(e[1][0] / size[0]) - j - l2) % size[1]

                        if e[1][1]:
                            bl1, bl2 = 0, 0
                        else:
                            # The border edges are considered broken so that they become reflexive
                            if x + l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                                bl1, bl2 = 0, 0
                            else:
                                bl1, bl2 = l1, l2

                        m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + (x + bl1) * size[1] + (y + bl2)
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield m, n, 1

            rdd = self._spark_context.range(
                size_xy
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
            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        # The border edges are considered broken so that they become reflexive
                        if x + l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                            bl1, bl2 = 0, 0
                        else:
                            bl1, bl2 = l1, l2

                        m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + (x + bl1) * size[1] + (y + bl2)
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield m, n, 1

            rdd = self._spark_context.range(
                size_xy
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

        operator = Operator(rdd, shape, coord_format=coord_format).materialize(storage_level)

        if self._broken_links_probability:
            broken_links.unpersist()

        self._profile(operator, initial_time)

        return operator
