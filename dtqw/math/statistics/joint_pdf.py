import math

from dtqw.math.statistics.pdf import PDF

__all__ = ['JointPDF']


class JointPDF(PDF):
    """Class for probability distribution function of the entire quantum system."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """
        Build an object for probability distribution function of the entire quantum system.

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

    def sum(self):
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
            ind = self._num_particles
        elif self._mesh.is_2d():
            ind = self._num_particles * 2
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
            ind = self._num_particles
        elif self._mesh.is_2d():
            ind = 2 * self._num_particles
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
            mean = self.expected_value()

        def _map(m):
            v = 1

            for i in range(0, ind, ndim):
                for d in range(ndim):
                    v *= m[i + d] - mesh_size[d]

            return m[ind] * v ** 2

        data_type = self._data_type()

        return self.data.filter(
            lambda m: m[ind] != data_type
        ).map(
            _map
        ).reduce(
            lambda a, b: a + b
        ) - mean
