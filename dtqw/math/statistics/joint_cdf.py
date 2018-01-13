from dtqw.math.statistics.cdf import CDF

__all__ = ['JointCDF']


class JointCDF(CDF):
    """Class for cumulative density function of the entire quantum system."""

    def __init__(self, spark_context, rdd, shape, mesh, num_particles):
        """
        Build an object for cumulative density function of the entire quantum system.

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

    def sum(self, round_precision=10):
        """
        Sum the values of this CDF.

        Parameters
        ----------
        round_precision : int, optional
            The precision used to round the value. Default is 10 decimal digits.

        Returns
        -------
        float
            The sum of the probabilities.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ind = self._num_particles
        elif self._mesh.is_2d():
            ind = self._num_particles * 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, round_precision)

    def expected_value(self, round_precision=10):
        """
        Calculate the expected value of this CDF.

        Parameters
        ----------
        round_precision : int, optional
            The precision used to round the value. Default is 10 decimal digits.

        Returns
        -------
        float
            The expected value.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ndim = 1
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        def _map(m):
            v = 1

            for i in range(0, ind, ndim):
                for d in range(ndim):
                    v *= m[i + d] - mesh_size[d]

            return m[ind] * v

        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, round_precision)

    def variance(self, mean=None, round_precision=10):
        """
        Calculate the variance of this CDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this CDF. When None is passed as argument, the mean is calculated.
        round_precision : int, optional
            The precision used to round the value. Default is 10 decimal digits.

        Returns
        -------
        float
            The variance.

        Raises
        ------
        NotImplementedError

        """
        if self._mesh.is_1d():
            ndim = 1
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size / 2), 1)
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * self._num_particles
            mesh_size = (int(self._mesh.size[0] / 2), int(self._mesh.size[1] / 2))
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        if mean is None:
            mean = self.expected_value(round_precision)

        def _map(m):
            v = 1

            for i in range(0, ind, ndim):
                for d in range(ndim):
                    v *= m[i + d] - mesh_size[d]

            return m[ind] * v ** 2

        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean

        return round(n, round_precision)
