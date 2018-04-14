import numpy as np

from pyspark import StorageLevel

from dtqw.mesh.mesh import Mesh
from dtqw.utils.utils import Utils

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    """Top-level class for 2-dimensional Meshes."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a top-level 2-dimensional Mesh object.

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

    def _define_num_edges(self, size):
        raise NotImplementedError

    def filename(self):
        if self._broken_links:
            probability = self._broken_links.probability
        else:
            probability = 0.0

        return "{}_{}-{}_{}".format(
            self.to_string(), self._size[0], self._size[1], probability
        )

    def axis(self):
        return np.meshgrid(
            range(self._size[0]),
            range(self._size[1]),
            indexing='ij'
        )

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

    def create_operator(self, num_partitions,
                        coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the mesh operator.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError
