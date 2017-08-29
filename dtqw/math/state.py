import math
from datetime import datetime
from pyspark import StorageLevel
from dtqw.math.pdf import PDF, is_pdf
from dtqw.mesh.mesh import is_mesh
from dtqw.math.matrix import Matrix

__all__ = ['State', 'is_state']


class State(Matrix):
    def __init__(self, spark_context, rdd, shape, mesh, num_particles, log_filename='./log.txt'):
        super().__init__(spark_context, rdd, shape, log_filename)

        if not is_mesh(mesh):
            self._logger.error('Mesh instance expected, not "{}"'.format(type(mesh)))
            raise TypeError('Mesh instance expected, not "{}"'.format(type(mesh)))

        self._mesh = mesh

        self._num_particles = num_particles

    @property
    def mesh(self):
        return self._mesh

    @property
    def num_particles(self):
        return self._num_particles

    def is_unitary(self, round_precision=10):
        n = self.data.filter(
            lambda m: m[1] != complex()
        ).map(
            lambda m: m[1].real ** 2 + m[1].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return round(math.sqrt(n), round_precision) == 1.0

    def kron(self, other_broadcast, other_shape):
        '''
        if not is_state(other):
            self._logger.error('Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(other)))

        o_shape = other.shape
        shape = (self._shape[0] * o_shape[0], 1)

        rdd = self.data.cartesian(
            other.data
        ).map(
            lambda m: (m[0][0] * o_shape[0] + m[1][0], m[0][1] * m[1][1])
        )
        '''

        shape = (self._shape[0] * other_shape[0], 1)

        def __map(m):
            for i in other_broadcast.value:
                yield (m[0] * other_shape[0] + i[0], m[1] * i[1])

        rdd = self.data.flatMap(
            __map
        )

        return State(self._spark_context, rdd, shape, self._mesh, self._num_particles, self._logger.filename)

    def full_measurement(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Measuring the state of the system...")

        t1 = datetime.now()

        coin_size = 2

        if self._mesh.is_1d():
            ndim = 1
            num_p = self._num_particles
            ind = ndim * num_p
            size = self._mesh.size
            cs_size = coin_size * size
            dims = [size for p in range(ind)]

            if self._num_particles == 1:
                dims.append(1)

            shape = tuple(dims)

            def __map(m):
                a = []

                for p in range(num_p):
                    a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)

                return tuple(a), (abs(m[1]) ** 2).real

            def __unmap(m):
                a = []

                for p in range(num_p):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        elif self._mesh.is_2d():
            ndim = 2
            num_p = self._num_particles
            ind = ndim * num_p
            dims = []

            for p in range(0, ind, ndim):
                dims.append(self._mesh.size[0])
                dims.append(self._mesh.size[1])

            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            shape = tuple(dims)

            def __map(m):
                a = []

                for p in range(num_p):
                    a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                    a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)

                return tuple(a), (abs(m[1]) ** 2).real

            def __unmap(m):
                a = []

                for p in range(ind):
                    a.append(m[0][p])

                a.append(m[1])

                return tuple(a)
        else:
            self._logger.error("Mesh dimension not implemented")
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

        pdf = PDF(
            self._spark_context, rdd, shape, self._mesh, self._num_particles, self._logger.filename
        ).materialize(storage_level)

        self._logger.info("Checking if the probabilities sum one...")

        if pdf.sum(ind) != 1.0:
            self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        self._logger.info("Full measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

        return pdf

    def filtered_measurement(self, full_measurement, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Measuring the state of the system which the particles are at the same positions...")

        t1 = datetime.now()

        if not is_pdf(full_measurement):
            self._logger.error('PDF instance expected, not "{}"'.format(type(full_measurement)))
            raise TypeError('PDF instance expected, not "{}"'.format(type(full_measurement)))

        if self._mesh.is_1d():
            ndim = 1
            num_p = self._num_particles
            ind = ndim * num_p
            size = self._mesh.size
            shape = (size, 1)

            def __filter(m):
                for p in range(num_p):
                    if m[0] != m[p]:
                        return False
                return True

            def __map(m):
                return m[0], m[ind]
        elif self._mesh.is_2d():
            ndim = 2
            num_p = self._num_particles
            ind = ndim * num_p
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
            self._logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = full_measurement.data.filter(
            __filter
        ).map(
            __map
        )

        pdf = PDF(
            self._spark_context, rdd, shape, self._mesh, self._num_particles, self._logger.filename
        ).materialize(storage_level)

        self._logger.info("Filtered measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

        return pdf

    def _partial_measurement(self, particle, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Measuring the state of the system for particle {}...".format(particle + 1))

        t1 = datetime.now()

        coin_size = 2

        if self._mesh.is_1d():
            ind = 1
            num_p = self._num_particles
            size = self._mesh.size
            cs_size = coin_size * size
            shape = (size, 1)

            def __map(m):
                a = []

                for p in range(num_p):
                    a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)

                return a[particle], (abs(m[1]) ** 2).real

            def __unmap(m):
                return m
        elif self._mesh.is_2d():
            ind = 2
            num_p = self._num_particles
            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            shape = (size_x, size_y)

            def __map(m):
                a = []

                for p in range(num_p):
                    a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                    a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)

                return (a[particle], a[particle + 1]), (abs(m[1]) ** 2).real

            def __unmap(m):
                return m[0][0], m[0][1], m[1]
        else:
            self._logger.error("Mesh dimension not implemented")
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

        pdf = PDF(
            self._spark_context, rdd, shape, self._mesh, self._num_particles, self._logger.filename
        ).materialize(storage_level)

        self._logger.info("Checking if the probabilities sum one...")

        if pdf.sum(ind) != 1.0:
            self._logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        self._logger.info(
            "Partial measurement for particle {} was done in {}s".format(
                particle + 1, (datetime.now() - t1).total_seconds()
            )
        )

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

        return pdf

    def partial_measurements(self, particles, storage_level=StorageLevel.MEMORY_AND_DISK):
        return [self._partial_measurement(p, storage_level) for p in particles]


def is_state(obj):
    return isinstance(obj, State)
