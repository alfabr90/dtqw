import random
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.mesh.mesh import Mesh
from dtqw.linalg.matrix import Matrix

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    """Top level class for 1-dimensional Meshes."""

    def __init__(self, spark_context, size, bl_prob=None):
        """
        Build a top level 1-dimensional Mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int
            Size of the mesh.
        bl_prob : float, optional
            Probability of the occurences of broken links in the mesh.
        """
        super().__init__(spark_context, size, bl_prob)

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

    def filename(self):
        return "{}_{}_{}".format(self.to_string(), self._size, self._broken_links_probability)

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

    def create_operator(self, num_partitions,
                        coord_format=Matrix.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the mesh operator.

        Parameters
        ----------
        num_partitions : int
            The desired number of partitions for the RDD.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Matrix.CoordinateDefault.
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
        bl_prop = self._broken_links_probability

        if not bl_prop:
            return broadcast(self._spark_context, {})

        # An example of a 5x1 mesh:
        #
        # 00 O 01 O 02 O 03 O 04 O 05
        # ---------------------------
        #              x
        def __map(e):
            random.seed()
            return e, random.random() < bl_prop

        rdd = self._spark_context.range(
            size + 1
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        ).collectAsMap()

        return broadcast(self._spark_context, rdd)
