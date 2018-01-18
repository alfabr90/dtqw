import random

from pyspark import StorageLevel

from dtqw.mesh.mesh2d.mesh2d import Mesh2D
from dtqw.utils.utils import broadcast, CoordinateDefault

__all__ = ['Natural']


class Natural(Mesh2D):
    """Top-level class for Natural Meshes."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a top-level Natural Mesh object.

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

    def generate_broken_links(self):
        """
        Yield broken edges for the mesh based on its probability to have a broken link.

        Returns
        -------
        RDD
            The RDD which keys are the numbered edges that are broken.

        """
        size = self._size
        size_bl = size[0] * size[1] + size[0] * size[1]

        bl_prop = self._broken_links_probability

        if not bl_prop:
            return self._spark_context.emptyRDD()

        # An example of a 5x5 diagonal mesh:
        #
        #   25   30   35   40   45    |
        # 20 O 21 O 22 O 23 O 24 O 20 |
        #   29   34   39   44   50    |
        # 15 O 16 O 17 O 18 O 19 O 15 |
        #   28   33   38   43   49    |
        # 10 O 11 O 12 O 13 O 14 O 10 | y
        #   27   32   37   42   48    |
        # 05 O 06 O 07 O 08 O 09 O 05 |
        #   26   31   36   41   47    |
        # 00 O 01 O 02 O 03 O 04 O 00 |
        #   25   30   35   40   45    |
        # ---------------------------
        #              x
        def __map(e):
            random.seed()
            return e, random.random() < bl_prop

        return self._spark_context.range(
            size_bl
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        )
