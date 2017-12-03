from datetime import datetime
from pyspark import StorageLevel
from dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from dtqw.linalg.operator import Operator

__all__ = ['TorusDiagonal']


class TorusDiagonal(Diagonal):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self._size = self._define_size(size)

    def title(self):
        return 'Diagonal Torus'

    def check_steps(self, steps):
        return True

    def create_operator(self, num_partitions, mul_format=True, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the shift operator for the walk.

        Parameters
        ----------
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
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        coin_size = 2
        size = self._size
        size_xy = size[0] * size[1]
        shape = (coin_size * coin_size * size_xy, coin_size * coin_size * size_xy)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        e = (y + 1 - j) * (size[0] + 1) + x + 1 - i

                        if e in bl_broad.value:
                            bl1, bl2 = 0, 0
                        else:
                            bl1, bl2 = l1, l2

                        m = ((i + bl1) * coin_size + (j + bl2)) * size_xy + \
                            ((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])
                        n = ((1 - i) * coin_size + (1 - j)) * size_xy + x * size[1] + y

                        yield (m, n, 1)
        else:
            def __map(xy):
                x = xy % size[0]
                y = int(xy / size[0])

                for i in range(coin_size):
                    l1 = (-1) ** i
                    for j in range(coin_size):
                        l2 = (-1) ** j

                        m = (i * coin_size + j) * size_xy + ((x + l1) % size[0]) * size[1] + ((y + l2) % size[1])
                        n = (i * coin_size + j) * size_xy + x * size[1] + y

                        yield (m, n, 1)

        rdd = self._spark_context.range(
            size_xy
        ).flatMap(
            __map
        )

        if mul_format:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            )

        rdd = rdd.partitionBy(
            numPartitions=num_partitions
        )

        operator = Operator(self._spark_context, rdd, shape).materialize(storage_level)

        if self._broken_links_probability:
            bl_broad.unpersist()

        self._profile(operator, initial_time)

        return operator
