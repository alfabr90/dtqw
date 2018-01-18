from datetime import datetime

from pyspark import StorageLevel

from dtqw.mesh.mesh2d.natural.natural import Natural
from dtqw.math.operator import Operator
from dtqw.utils.utils import CoordinateDefault, CoordinateMultiplier, CoordinateMultiplicand

__all__ = ['BoxNatural']


class BoxNatural(Natural):
    """Class for Natural Box mesh."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a Natural Box mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : tuple
            Size of the mesh.
        bl_prob : float, optional
            Probability of the occurences of broken links in the mesh.
        """
        super().__init__(spark_context, size, bl_prob)

        self._size = self._define_size(size)

    def title(self):
        return 'Natural Box'

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
            Default value is Operator.CoordinateDefault.
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
        size_bl = size[0] * size[1] + size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links_probability:
            broken_links = self.generate_broken_links()

            def __map(e):
                """e = (edge, (edge, broken or not))"""
                for i in range(coin_size):
                    l = (-1) ** i

                    # Finding the correspondent x,y coordinates of the vertex from the edge number
                    if e[1][0] >= size[0] * size[1]:
                        j = i
                        delta = int(not (i ^ j))
                        x = int((e[1][0] - size[0] * size[1]) / size[0])
                        y = ((e[1][0] - size[0] * size[1]) % size[1] - i - l) % size[1]
                    else:
                        j = int(not i)
                        delta = int(not (i ^ j))
                        x = (e[1][0] % size[0] - i - l) % size[0]
                        y = int(e[1][0] / size[0])

                    pos1 = x + l * (1 - delta)
                    pos2 = y + l * delta

                    if e[1][1]:
                        bl = 0
                    else:
                        # The border edges are considered broken so that they become reflexive
                        if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                            bl = 0
                        else:
                            bl = l

                    m = ((i + bl) * coin_size + (abs(j + bl) % coin_size)) * size_xy + \
                        (x + bl * (1 - delta)) * size[1] + (y + bl * delta)
                    n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                    yield m, n, 1

            rdd = self._spark_context.range(
                size_bl
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
                    l = (-1) ** i
                    for j in range(coin_size):
                        delta = int(not (i ^ j))

                        pos1 = x + l * (1 - delta)
                        pos2 = y + l * delta

                        # The border edges are considered broken so that they become reflexive
                        if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                            bl = 0
                        else:
                            bl = l

                        m = ((i + bl) * coin_size + (abs(j + bl) % coin_size)) * size_xy + \
                            (x + bl * (1 - delta)) * size[1] + (y + bl * delta)
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

        operator = Operator(self._spark_context, rdd, shape, coord_format).materialize(storage_level)

        if self._broken_links_probability:
            broken_links.unpersist()

        self._profile(operator, initial_time)

        return operator
