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

    def broken_links(self):
        """
        Yield broken edges for the mesh based on its probability to have a broken link.

        Returns
        -------
        Broadcast
            A Spark's broadcast variable containing a dict which keys are the numbered edges that are broken.

        """
        size = self._size
        size_bl = (size[0] + 1) * (size[1] + 1)

        bl_prop = self._broken_links_probability

        if not bl_prop:
            return broadcast(self._spark_context, {})

        # An example of topology with a 5x5 diagonal mesh:
        #
        # 30 31 32 33 34 35 |
        #   O  O  O  O  O   |
        # 24 25 26 27 28 29 |
        #   O  O  O  O  O   |
        # 18 19 20 21 22 23 |
        #   O  O  O  O  O   | y
        # 12 13 14 15 16 17 |
        #   O  O  O  O  O   |
        # 06 07 08 09 10 11 |
        #   O  O  O  O  O   |
        # 00 01 02 03 04 05 |
        # -----------------
        #         x
        def __map(e):
            random.seed()
            return e, random.random() < bl_prop

        rdd = self._spark_context.range(
            size_bl
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        ).collectAsMap()

        return broadcast(self._spark_context, rdd)
