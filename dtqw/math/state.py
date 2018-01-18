import math
from datetime import datetime

from pyspark import StorageLevel

from dtqw.math.base import Base
from dtqw.math.statistics.cdf import is_cdf
from dtqw.math.statistics.joint_cdf import JointCDF
from dtqw.math.statistics.filtered_cdf import FilteredCDF
from dtqw.math.statistics.marginal_cdf import MarginalCDF
from dtqw.mesh.mesh import is_mesh

__all__ = ['State', 'is_state']


class State(Base):
    """Class for the system state."""

    def __init__(self, spark_context, rdd, shape, mesh, num_particles):
        """
        Build a State object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this operator object. Must be a 2-dimensional tuple.
        mesh : Mesh
            The mesh where the particles will walk on.
        num_particles : int
            The number of particles present in the walk.

        """
        if not is_mesh(mesh):
            # self.logger.error('Mesh instance expected, not "{}"'.format(type(mesh)))
            raise TypeError('mesh instance expected, not "{}"'.format(type(mesh)))

        super().__init__(spark_context, rdd, shape)

        self._mesh = mesh
        self._num_particles = num_particles

    @property
    def mesh(self):
        return self._mesh

    @property
    def num_particles(self):
        return self._num_particles

    def kron(self, other):
        """
        Perform a tensor (Kronecker) product with another system state.

        Parameters
        ----------
        other : :obj:State
            The other system state.

        Returns
        -------
        :obj:State
            The resulting state.

        """
        if not is_state(other):
            if self._logger:
                self._logger.error('State instance expected, not "{}"'.format(type(other)))
            raise TypeError('State instance expected, not "{}"'.format(type(other)))

        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], 1)

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.map(
            lambda m: (0, m)
        ).join(
            other.data.map(
                lambda m: (0, m)
            ),
            numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0], m[1][1])
        )

        # rdd = self.data.cartesian(
        #     other.data
        # )

        rdd = rdd.map(
            lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * m[1][1])
        )

        return State(self._spark_context, rdd, new_shape, self._mesh, self._num_particles)

    def norm(self):
        """
        Calculate the norm of this quantum state.

        Returns
        -------
        float
            The norm of this quantum state.

        """
        n = self.data.filter(
            lambda m: m[1] != complex()
        ).map(
            lambda m: m[1].real ** 2 + m[1].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def full_measurement(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform the measurement of the entire system state.

        Parameters
        ----------
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        :obj:JointCDF
            The CDF of the entire system.

        Raises
        ------
        NotImplementedError
        ValueError

        """
        if self._logger:
            self._logger.info("measuring the state of the system...")

        t1 = datetime.now()

        coin_size = 2

        if self._mesh.is_1d():
            ndim = 1
            num_particles = self._num_particles
            ind = ndim * num_particles
            size = self._mesh.size
            cs_size = coin_size * size
            dims = [size for p in range(ind)]

            if self._num_particles == 1:
                dims.append(1)

            shape = tuple(dims)

            def __map(m):
                x = []

                for p in range(num_particles):
                    x.append(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size)

                return tuple(x), (abs(m[1]) ** 2).real

            def __unmap(m):
                a = []

                for p in range(num_particles):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        elif self._mesh.is_2d():
            ndim = 2
            num_particles = self._num_particles
            ind = ndim * num_particles
            dims = []

            for p in range(0, ind, ndim):
                dims.append(self._mesh.size[0])
                dims.append(self._mesh.size[1])

            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            cs_size_xy = cs_size_x * cs_size_y
            shape = tuple(dims)

            def __map(m):
                xy = []

                for p in range(num_particles):
                    xy.append(
                        (
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x,
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y
                        )
                    )

                return tuple(xy), (abs(m[1]) ** 2).real

            def __unmap(m):
                xy = []

                for p in range(num_particles):
                    xy.append(m[0][p][0])
                    xy.append(m[0][p][1])

                xy.append(m[1])

                return tuple(xy)
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self.data.filter(
            lambda m: m[1] != complex()
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=self.data.getNumPartitions()
        ).map(
            __unmap
        )

        cdf = JointCDF(self._spark_context, rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        if self._logger:
            self._logger.info("checking if the probabilities sum one...")

        if round(cdf.sum(), 10) != 1.0:
            if self._logger:
                self._logger.error("CDFs must sum one")
            raise ValueError("CDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_cdf('fullMeasurement', cdf, (datetime.now() - t1).total_seconds())

            if self._logger:
                self._logger.info("full measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "CDF with full measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return cdf

    def filtered_measurement(self, full_measurement, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Filter the measurement of the entire system by checking when
        all particles are located at the same site of the mesh.

        Parameters
        ----------
        full_measurement : :obj:CDF
            The measurement of the entire system.
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        :obj:FilteredCDF
            The CDF of the system when all particles are located at the same site.

        Raises
        ------
        NotImplementedError

        """
        if self._logger:
            self._logger.info("measuring the state of the system which the particles are at the same positions...")

        t1 = datetime.now()

        if not is_cdf(full_measurement):
            if self._logger:
                self._logger.error('CDF instance expected, not "{}"'.format(type(full_measurement)))
            raise TypeError('CDF instance expected, not "{}"'.format(type(full_measurement)))

        if self._mesh.is_1d():
            ndim = 1
            num_particles = self._num_particles
            ind = ndim * num_particles
            size = self._mesh.size
            shape = (size, 1)

            def __filter(m):
                for p in range(num_particles):
                    if m[0] != m[p]:
                        return False
                return True

            def __map(m):
                return m[0], m[ind]
        elif self._mesh.is_2d():
            ndim = 2
            num_particles = self._num_particles
            ind = ndim * num_particles
            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            shape = (size_x, size_y)

            def __filter(m):
                for p in range(0, ind, ndim):
                    if m[0] != m[p] or m[1] != m[p + 1]:
                        return False
                return True

            def __map(m):
                return m[0], m[1], m[ind]
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = full_measurement.data.filter(
            __filter
        ).map(
            __map
        )

        cdf = FilteredCDF(self._spark_context, rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_cdf('filteredMeasurement', cdf, (datetime.now() - t1).total_seconds())

            if self._logger:
                self._logger.info("filtered measurement was done in {}s".format(info['buildingTime']))
                self._logger.info(
                    "CDF with filtered measurement is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return cdf

    def partial_measurement(self, particle, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform the partial measurement of a particle of the system state.

        Parameters
        ----------
        particle : int
            The desired particle to be measured. The particle number starts by 0.
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        :obj:MarginalCDF
            The CDF of each particle.

        Raises
        ------
        NotImplementedError
        ValueError

        """
        if self._logger:
            self._logger.info("measuring the state of the system for particle {}...".format(particle + 1))

        t1 = datetime.now()

        coin_size = 2

        if self._mesh.is_1d():
            num_particles = self._num_particles
            size = self._mesh.size
            cs_size = coin_size * size
            shape = (size, 1)

            def __map(m):
                x = []

                for p in range(num_particles):
                    x.append(int(m[0] / (cs_size ** (num_particles - 1 - p))) % size)

                return x[particle], (abs(m[1]) ** 2).real

            def __unmap(m):
                return m
        elif self._mesh.is_2d():
            num_particles = self._num_particles
            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            cs_size_xy = cs_size_x * cs_size_y
            shape = (size_x, size_y)

            def __map(m):
                xy = []

                for p in range(num_particles):
                    xy.append(
                        (
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x,
                            int(m[0] / (cs_size_xy ** (num_particles - 1 - p))) % size_y
                        )
                    )

                return xy[particle], (abs(m[1]) ** 2).real

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self.data.filter(
            lambda m: m[1] != complex()
        ).map(
            __map
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=self.data.getNumPartitions()
        ).map(
            __unmap
        )

        cdf = MarginalCDF(self._spark_context, rdd, shape, self._mesh, self._num_particles).materialize(storage_level)

        if self._logger:
            self._logger.info("checking if the probabilities sum one...")

        if round(cdf.sum(), 10) != 1.0:
            if self._logger:
                self._logger.error("CDFs must sum one")
            raise ValueError("CDFs must sum one")

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_cdf(
                'partialMeasurementParticle{}'.format(particle + 1), cdf, (datetime.now() - t1).total_seconds()
            )

            if self._logger:
                self._logger.info("partial measurement for particle {} was done in {}s".format(
                        particle + 1, info['buildingTime'])
                )
                self._logger.info(
                    "CDF with partial measurements for particle {} "
                    "are consuming {} bytes in memory and {} bytes in disk".format(
                        particle + 1, info['memoryUsed'], info['diskUsed']
                    )
                )

            self._profiler.log_rdd(app_id=app_id)

        return cdf

    def partial_measurements(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform the partial measurement of each particle of the system state.

        Parameters
        ----------
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        tuple
            A tuple containing the CDF of each particle.

        """
        return [self.partial_measurement(p, storage_level) for p in range(self._num_particles)]

    def measure(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform the measurement of the system state.

        If the state is composed by only one particle, the full measurement of the
        system is performed and returned. In other cases, the measurement process will return a tuple containing
        the full measurement, the filtered measurement - probabilities of each mesh site
        with all particles located at - and the partial measurement of each particle.

        Parameters
        ----------
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        :obj:CDF or tuple
            CDF if the system is composed by only one particle, tuple otherwise.

        """
        if self._num_particles == 1:
            return self.full_measurement(storage_level)
        else:
            full_measurement = self.full_measurement(storage_level)
            filtered_measurement = self.filtered_measurement(full_measurement, storage_level)
            partial_measurements = self.partial_measurements(storage_level)

            return full_measurement, filtered_measurement, partial_measurements


def is_state(obj):
    """
    Check whether argument is a State object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a State object, False otherwise.

    """
    return isinstance(obj, State)
