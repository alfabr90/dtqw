import numpy as np

from pyspark import StorageLevel

from dtqw.mesh.mesh import Mesh
from dtqw.utils.utils import CoordinateDefault

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    """Top-level class for 2-dimensional Meshes."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a top-level 2-dimensional Mesh object.

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

    def _validate(self, size):
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                return False
        else:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            if self.logger:
                self.logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def filename(self):
        return "{}_{}-{}_{}".format(self.to_string(), self._size[0], self._size[1], self._broken_links_probability)

    def axis(self):
        return np.meshgrid(range(self._size[0]), range(self._size[1]))

    def is_1d(self):
        """
        Check if this is a 1-dimensional Mesh.

        Returns
        -------
        bool

        """
        return False

    def is_2d(self):
        """
        Check if this is a 2-dimensional Mesh.

        Returns
        -------
        bool

        """
        return True

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

    def generate_broken_links(self, num_partitions):
        """
        Yield broken edges for the mesh based on its probability to have a broken link.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.

        Raises
        ------
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
