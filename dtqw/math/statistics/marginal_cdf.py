import math

from dtqw.math.statistics.cdf import CDF

__all__ = ['MarginalCDF']


class MarginalCDF(CDF):
    """Class for cumulative density function of a particle in the quantum system."""

    def __init__(self, spark_context, rdd, shape, mesh, num_particles):
        """
        Build an object for cumulative density function of a particle in the quantum system.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        mesh : Mesh
            The mesh where the particles will walk on.
        num_particles : int
            The number of particles present in the walk.

        """
        super().__init__(spark_context, rdd, shape, mesh, num_particles)

    def sum(self):
        """
        Sum the values of this CDF.

        Returns
        -------
        float
            The sum of the probabilities.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = 1
        elif self._mesh.is_2d():
            ind = 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        return self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

    def norm(self):
        """
        Calculate the norm of this CDF.

        Returns
        -------
        float
            The norm of this CDF.

        """
        if self._mesh.is_1d():
            ind = 1
        elif self._mesh.is_2d():
            ind = 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            lambda m: m[ind].real ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def expected_value(self):
        """
        Calculate the expected value of this CDF.

        Returns
        -------
        float
            The expected value.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = 1
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ind = 2
            mesh_size = (int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def _map(m):
            v = 1

            for i in range(ind):
                v *= m[i] - mesh_size[i]

            return m[ind] * v

        return self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        )

    def variance(self, mean=None):
        """
        Calculate the variance of this CDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this CDF. When None is passed as argument, the mean is calculated.

        Returns
        -------
        float
            The variance.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = 1
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ind = 2
            mesh_size = (int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        if mean is None:
            mean = self.expected_value()

        def _map(m):
            v = 1

            for i in range(ind):
                v *= m[i] - mesh_size[i]

            return m[ind] * v ** 2

        return self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean
