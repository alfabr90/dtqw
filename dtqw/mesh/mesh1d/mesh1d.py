from pyspark import StorageLevel

from dtqw.mesh.mesh import Mesh
from dtqw.utils.utils import Utils

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    """Top-level class for 1-dimensional Meshes."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a top-level 1-dimensional Mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int
            Size of the mesh.
        broken_links : BrokenLinks, optional
            A BrokenLinks object.
        """
        super().__init__(spark_context, size, broken_links=broken_links)

    def _validate(self, size):
        if type(size) != int:
            return False
        elif size <= 0:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            if self.logger:
                self.logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def _define_num_edges(self, size):
        # The number of edges is the same of the size of the mesh.
        # For the already implemented types of mesh, the border edges are the same.
        #
        # An example of a 5x1 mesh:
        #
        # 00 O 01 O 02 O 03 O 04 O 00
        # ---------------------------
        #              x
        return self._define_size(size)

    def filename(self):
        if self._broken_links:
            probability = self._broken_links.probability
        else:
            probability = 0.0

        return "{}_{}_{}".format(
            self.to_string(), self._size, probability
        )

    def axis(self):
        return range(self._size)

    def is_1d(self):
        """
        Check if this is a 1-dimensional Mesh.

        Returns
        -------
        bool

        """
        return True

    def is_2d(self):
        """
        Check if this is a 2-dimensional Mesh.

        Returns
        -------
        bool

        """
        return False

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

    def create_operator(self, coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the mesh operator.

        Parameters
        ----------
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
