import math

from dtqw.math.statistics.pdf import PDF

__all__ = ['MarginalPDF']


class MarginalPDF(PDF):
    """Class for probability distribution function of a particle in the quantum system."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """
        Build an object for probability distribution function of a particle in the quantum system.

        Parameters
        ----------
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        mesh : Mesh
            The mesh where the particles will walk on.
        num_particles : int
            The number of particles present in the walk.

        """
        super().__init__(rdd, shape, mesh, num_particles)

    def sum_values(self):
        """
        Sum the values of this PDF.

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

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

    def norm(self):
        """
        Calculate the norm of this PDF.

        Returns
        -------
        float
            The norm of this PDF.

        """
        if self._mesh.is_1d():
            ind = 1
        elif self._mesh.is_2d():
            ind = 2
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        data_type = self._data_type()

        n = self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            lambda m: m[ind].real ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def expected_value(self):
        """
        Calculate the expected value of this PDF.

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

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        )

    def variance(self, mean=None):
        """
        Calculate the variance of this PDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this PDF. When None is passed as argument, the mean is calculated.

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

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean

    # def sum(self, other):
    #     if not is_pdf(other):
    #         if self._logger:
    #             self._logger.error('PDF instance expected, not "{}"'.format(type(other)))
    #         raise TypeError('PDF instance expected, not "{}"'.format(type(other)))
    #
    #     if len(self._shape) != len(other.shape):
    #         if self._logger:
    #             self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
    #         raise ValueError('incompatible shapes {} and {}'.format(self._shape, other.shape))
    #
    #     for i in len(self._shape):
    #         if self._shape[i] != other.shape[i]:
    #             if self._logger:
    #                 self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
    #             raise ValueError('incompatible shapes {} and {}'.format(self._shape, other.shape))
    #
    #     shape = self._shape
    #     num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())
    #
    #     num_particles = self._num_particles
    #
    #     if self._mesh.is_1d():
    #         def __map(m):
    #             x = []
    #
    #             for p in range(num_particles):
    #                 x.append(m[p])
    #
    #             return tuple(x), m[num_particles]
    #
    #         def __unmap(m):
    #             a = []
    #
    #             for p in range(num_particles):
    #                 a.append(m[0][p])
    #
    #             a.append(m[1])
    #
    #             return tuple(a)
    #     elif self._mesh.is_2d():
    #         ndim = 2
    #         ind = ndim * num_particles
    #
    #         def __map(m):
    #             xy = []
    #
    #             for p in range(0, ind, ndim):
    #                 xy.append(m[p])
    #
    #             return tuple(x), m[num_particles]
    #
    #         def __unmap(m):
    #             a = []
    #
    #             for p in range(0, ind, ndim):
    #                 a.append(m[0][p])
    #                 a.append(m[0][p + 1])
    #
    #             a.append(m[1])
    #
    #             return tuple(a)
    #     else:
    #         if self._logger:
    #             self._logger.error("mesh dimension not implemented")
    #         raise NotImplementedError("mesh dimension not implemented")
    #
    #     rdd = self.data.union(
    #         other.data
    #     ).map(
    #         __map
    #     ).reduceByKey(
    #         lambda a, b: a + b, numPartitions=num_partitions
    #     ).map(
    #         __unmap
    #     )
    #
    #     return MarginalPDF(rdd, shape, self._mesh, self._num_particles)
