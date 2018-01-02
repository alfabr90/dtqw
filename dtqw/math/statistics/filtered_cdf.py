from dtqw.math.statistics.cdf import CDF

__all__ = ['FilteredCDF']


class FilteredCDF(CDF):
    """Class for cumulative density function of the quantum system when the particles are at the same sites."""

    def __init__(self, spark_context, rdd, shape, mesh, num_particles):
        """
        Build an object for cumulative density function of the quantum system when the particles are at the same sites.

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
        Sum the probabilities of this CDF.

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

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def variance(self, mean=None, round_precision=10):
        """
        Calculate the variance of this CDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this CDF. When None is passed as argument, the mean is calculated.
        round_precision : int, optional
            The precision used to round the value. Default is 10 decimal digits.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError
