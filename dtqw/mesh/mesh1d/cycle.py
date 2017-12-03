from datetime import datetime
from pyspark import StorageLevel
from dtqw.mesh.mesh1d.mesh1d import Mesh1D
from dtqw.linalg.operator import Operator

__all__ = ['Cycle']


class Cycle(Mesh1D):
    def __init__(self, spark_context, size, bl_prob=None):
        super().__init__(spark_context, size, bl_prob)
        self.__size = self._define_size(size)

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
        shape = (coin_size * size, coin_size * size)

        if self._broken_links_probability:
            bl_broad = self.broken_links()

            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i

                    if bl_broad.value.get((x + i + l) % size) is not None:
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
