from datetime import datetime

from pyspark import StorageLevel

from dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from dtqw.math.operator import Operator
from dtqw.utils.utils import Utils

__all__ = ['BoxDiagonal']


class BoxDiagonal(Diagonal):
    """Class for Diagonal Box mesh."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a Diagonal Box mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : tuple
            Size of the mesh.
        broken_links : BrokenLinks, optional
            A BrokenLinks object.
        """
        super().__init__(spark_context, size, broken_links=broken_links)

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

        Raises
        ------
        ValueError

        """
        if self._logger:
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        coin_size = 2
        size = self._size
        num_edges = self._num_edges
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = Utils.getConf(self._spark_context, 'dtqw.mesh.brokenLinks.generationMode', default='broadcast')

            if generation_mode == 'rdd':
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
                    num_edges
                ).map(
                    lambda m: (m, m)
                ).leftOuterJoin(
                    broken_links
                ).flatMap(
                    __map
                )
            elif generation_mode == 'broadcast':
                def __map(e):
                    for i in range(coin_size):
                        l1 = (-1) ** i
                        for j in range(coin_size):
                            l2 = (-1) ** j

                            # Finding the correspondent x,y coordinates of the vertex from the edge number
                            x = (e % size[0] - i - l1) % size[0]
                            y = (int(e / size[0]) - j - l2) % size[1]

                            if e in broken_links.value:
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
                    num_edges
                ).flatMap(
                    __map
                )
            else:
                if self._logger:
                    self._logger.error("invalid broken links generation mode")
                raise ValueError("invalid broken links generation mode")
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
