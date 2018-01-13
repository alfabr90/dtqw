import random

from pyspark import StorageLevel

from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.utils.utils import broadcast, CoordinateDefault

__all__ = ['Diagonal']


class Diagonal(Mesh2D):
    """Top-level class for Diagonal Meshes."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a top-level Diagonal Mesh object.

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

    def check_steps(self, steps):
        """
        Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def create_operator(self, num_partitions,
                        coord_format=CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the mesh operator.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Operator.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def generate_broken_links(self, num_partitions):
        """
        Yield broken edges for the mesh based on its probability to have a broken link.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.

        Returns
        -------
        RDD
            The RDD which keys are the numbered edges that are broken.

        """
        size = self._size
        size_bl = size[0] * size[1]

        bl_prop = self._broken_links_probability

        if not bl_prop:
            return self._spark_context.emptyRDD()

        # An example of a 5x5 diagonal mesh:
        #
        # 00 01 01 03 04 00 |
        #   O  O  O  O  O   |
        # 20 21 22 23 24 20 |
        #   O  O  O  O  O   |
        # 15 16 17 18 19 20 |
        #   O  O  O  O  O   | y
        # 10 11 12 13 14 10 |
        #   O  O  O  O  O   |
        # 05 06 07 08 09 05 |
        #   O  O  O  O  O   |
        # 00 01 02 03 04 00 |
        # -----------------
        #         x
        def __map(e):
            random.seed()
            return e, random.random() < bl_prop

        return self._spark_context.range(
            size_bl
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        ).partitionBy(
            numPartitions=num_partitions
        )
