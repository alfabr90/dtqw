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

    def expected_value(self, ind, round_precision=10):
        def _map(m):
            v = 1

            for i in range(ind):
                v *= m[i]

            return m[ind] * v

        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) / self._shape[0]

        return round(n, round_precision)
