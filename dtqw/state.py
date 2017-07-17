__all__ = ['State', 'is_state']


import os
import shutil
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import fileinput as fi

from datetime import datetime
from glob import glob
from pyspark import RDD, StorageLevel

from .pdf import *
from .logger import *
from .metrics import *
from .utils import is_shape, convert_sparse, get_size_of, get_tmp_path, remove_tmp_path, broadcast


class State:
    def __init__(self, arg1, spark_context, mesh, shape=None, num_particles=1, rdd_path=None, log_filename='log.txt'):
        self.__spark_context = spark_context
        self.__mesh = mesh
        self.__num_particles = num_particles
        self.__format = None
        self.__value_type = complex
        self.__memory_usage = None
        self.__rdd_path = None
        self.__logger = Logger(__name__, log_filename)
        self.__metrics = Metrics(log_filename=log_filename)

        self.data = None
        self.shape = None

        if shape is not None:
            if not is_shape(shape):
                raise ValueError

        if type(arg1) == str:
            self.__from_path(arg1, shape)
        elif isinstance(arg1, RDD):
            self.__from_rdd(arg1, shape, rdd_path)
        elif isinstance(arg1, np.ndarray):
            self.__from_dense(arg1)
        elif sp.isspmatrix(arg1):
            self.__from_sparse(arg1)
        else:
            raise TypeError

    @property
    def spark_context(self):
        return self.__spark_context

    @property
    def mesh(self):
        return self.__mesh

    @property
    def num_particles(self):
        return self.__num_particles

    @property
    def format(self):
        return self.__format

    @property
    def value_type(self):
        return self.__value_type

    @property
    def memory_usage(self):
        return self.__memory_usage

    @property
    def rdd_path(self):
        return self.__rdd_path

    def __from_path(self, path, shape):
        self.data = path
        self.shape = shape
        self.__format = 'path'
        self.__memory_usage = self.__get_bytes()

    def __from_rdd(self, rdd, shape, rdd_path):
        self.data = rdd
        self.shape = shape
        self.__format = 'rdd'
        self.__rdd_path = rdd_path
        self.__memory_usage = self.__get_bytes()

    def __from_dense(self, dense):
        self.data = dense
        self.shape = dense.shape
        self.__format = 'dense'
        self.__memory_usage = self.__get_bytes()

    def __from_sparse(self, sparse):
        self.data = sparse
        self.shape = sparse.shape
        self.__format = 'sparse'
        self.__memory_usage = self.__get_bytes()

    def __get_bytes(self):
        return get_size_of(self.data)

    def is_path(self):
        return self.__format == 'path'

    def is_rdd(self):
        return self.__format == 'rdd'

    def is_dense(self):
        return self.__format == 'dense'

    def is_sparse(self):
        return self.__format == 'sparse'

    def is_unitary(self, round_precision=10):
        if self.is_path():
            n = self.data.map(
                lambda m: complex(m.split()[2])
            ).filter(
                lambda m: m != complex()
            ).map(
                lambda m: math.sqrt(m.real ** 2 + m.imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_rdd():
            n = self.data.filter(
                lambda m: m[2] != complex()
            ).map(
                lambda m: math.sqrt(m[2].real ** 2 + m[2].imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_dense():
            return round(np.linalg.norm(self.data), round_precision) == 1.0
        elif self.is_sparse():
            return round(splinalg.norm(self.data), round_precision) == 1.0
        else:
            # TODO
            raise NotImplementedError

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            if not self.data.is_cached:
                self.data.persist(storage_level)

    def unpersist(self):
        if self.is_rdd():
            if self.data is not None:
                self.data.unpersist()

    def destroy(self):
        self.unpersist()

        if self.is_path():
            remove_tmp_path(self.data)
        elif self.is_rdd():
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None

        self.data = None

    def copy(self):
        if self.is_path():
            path = get_tmp_path()

            if os.path.isdir(self.data):
                os.mkdir(path)

                for i in os.listdir(self.data):
                    shutil.copy(os.path.join(self.data, i), path)
            else:
                shutil.copy(self.data, path)

            return State(
                path,
                self.__spark_context,
                self.__mesh,
                self.shape,
                self.__num_particles,
                log_filename=self.__logger.filename
            )
        elif self.is_rdd():
            state = self.to_path(self.data.getNumPartitions(), True)
            return state.to_rdd(self.data.getNumPartitions())
        elif self.is_dense():
            return State(
                self.data.copy(),
                self.__spark_context,
                self.__mesh,
                self.__num_particles,
                log_filename=self.__logger.filename
            )
        elif self.is_sparse():
            return State(
                self.data.copy(),
                self.__spark_context,
                self.__mesh,
                self.__num_particles,
                log_filename=self.__logger.filename
            )

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            if not self.data.is_cached:
                self.persist(storage_level)
            self.data.count()

    def clear_rdd_path(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None

    def to_path(self, min_partitions=8, copy=False):
        if self.is_path():
            if copy:
                return self.copy()
            else:
                return self
        else:
            buffer_size = 8196

            if self.is_rdd():
                path = get_tmp_path()
                self.data.map(
                    lambda m: "{} {} {}".format(m[0], m[1], m[2])
                ).saveAsTextFile(path)
            elif self.is_dense():
                ind = self.data.nonzero()

                path = get_tmp_path()
                os.mkdir(path)

                files = [open(path + "/part-" + str(i), 'w', buffer_size) for i in range(min_partitions)]

                f = 0

                for i in range(len(ind[0])):
                    files[f].write("{} {} {}\n".format(ind[0][i], ind[1][i], self.data[ind[0][i], ind[1][i]]))
                    f = (f + 1) % min_partitions

                for f in files:
                    f.close()

                del ind
            elif self.is_sparse():
                ind = sp.find(self.data)

                path = get_tmp_path()
                os.mkdir(path)

                files = [open(path + "/part-" + str(i), 'w', buffer_size) for i in range(min_partitions)]

                f = 0

                for i in range(len(ind[2])):
                    files[f].write("{} {} {}\n".format(ind[0][i], ind[1][i], ind[2][i]))
                    f = (f + 1) % min_partitions

                for f in files:
                    f.close()

                del ind
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return State(
                    path,
                    self.__spark_context,
                    self.__mesh,
                    self.shape,
                    self.__num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                self.destroy()
                self.data = path
                self.__format = 'path'
                self.__memory_usage = get_size_of(path)
                return self

    def to_rdd(self, min_partitions=8, copy=False):
        if self.is_rdd():
            if copy:
                return self.copy()
            else:
                return self
        else:
            value_type = complex

            oper = self.to_path(min_partitions, copy)

            def __map(m):
                a = m.split()
                return int(a[0]), int(a[1]), value_type(a[2])

            rdd = self.__spark_context.textFile(
                oper.data, minPartitions=min_partitions
            ).map(
                __map
            ).filter(
                lambda m: m[2] != value_type()
            )

            if copy:
                return State(
                    rdd,
                    self.__spark_context,
                    self.__mesh,
                    self.shape,
                    self.__num_particles,
                    rdd_path=oper.data,
                    log_filename=self.__logger.filename
                )
            else:
                self.data = rdd
                self.__rdd_path = oper.data
                self.__format = 'rdd'
                self.__memory_usage = get_size_of(rdd)
                return self

    def to_dense(self, copy=False):
        if self.is_dense():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_path():
                dense = np.zeros(self.shape, dtype=complex)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        dense[int(l[0]), int(l[1])] += complex(l[2])
            elif self.is_rdd():
                # '''
                dense = np.zeros(self.shape, dtype=complex)

                for i in self.data.collect():
                    dense[i[0], i[1]] += i[2]
                '''
                path = get_tmp_path()

                self.data.map(
                    lambda m: "{} {} {}".format(m[0], m[1], m[2])
                ).saveAsTextFile(path)

                dense = np.zeros(self.shape, dtype=complex)

                with fi.input(files=glob(path + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        dense[int(l[0]), int(l[1])] += complex(l[2])

                remove_tmp_path(path)
                '''
            elif self.is_sparse():
                dense = self.data.toarray()
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return State(
                    dense,
                    self.__spark_context,
                    self.__mesh,
                    num_particles=self.__num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                self.destroy()
                self.data = dense
                self.__format = 'dense'
                self.__memory_usage = get_size_of(dense)
                return self

    def to_sparse(self, format='csr', copy=False):
        if self.is_sparse():
            if copy:
                return State(
                    convert_sparse(self.data.copy(), format),
                    self.__spark_context,
                    self.__mesh,
                    num_particles=self.__num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                return self
        else:
            if self.is_path():
                sparse = sp.dok_matrix(self.shape, dtype=complex)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        sparse[int(l[0]), int(l[1])] = complex(l[2])

                sparse = convert_sparse(sparse, format)
            elif self.is_rdd():
                # '''
                shape = self.shape

                def __map(m):
                    k = sp.dok_matrix(shape, dtype=complex)
                    k[m[0], m[1]] = m[2]
                    return k.tocsc()

                sparse = convert_sparse(
                    self.data.filter(
                        lambda m: m[2] != complex()
                    ).map(
                        __map
                    ).reduce(
                        lambda a, b: a + b
                    ),
                    format
                )
                '''
                path = get_tmp_path()

                self.data.map(
                    lambda m: "{} {} {}".format(m[0], m[1], m[2])
                ).saveAsTextFile(path)

                sparse = sp.dok_matrix(self.shape, dtype=complex)

                with fi.input(files=glob(path + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        sparse[int(l[0]), int(l[1])] += complex(l[2])

                remove_tmp_path(path)

                sparse = convert_sparse(sparse, format)
                '''
            elif self.is_dense():
                sparse = convert_sparse(sp.coo_matrix(self.data), format)
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return State(
                    sparse,
                    self.__spark_context,
                    self.__mesh,
                    num_particles=self.__num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                self.destroy()
                self.data = sparse
                self.__format = 'sparse'
                self.__memory_usage = get_size_of(sparse)
                return self

    def kron(self, other, min_partitions=8):
        if not is_state(other):
            raise TypeError('State instance expected (not "{}")'.format(type(other)))

        value_type = complex
        spark_context = self.__spark_context
        mesh = self.__mesh
        num_particles = self.__num_particles
        s_shape = self.shape
        o_shape = other.shape
        shape = (self.shape[0] * other.shape[0], self.shape[1] * other.shape[1])

        if self.is_sparse():
            if other.is_dense():
                return State(
                    sp.kron(self.data, sp.coo_matrix(other.data)),
                    spark_context,
                    mesh,
                    num_particles=num_particles,
                    log_filename=self.__logger.filename
                )
            elif other.is_sparse():
                return State(
                    sp.kron(self.data, other.data),
                    spark_context,
                    mesh,
                    num_particles=num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                # TODO
                raise NotImplementedError
        elif self.is_rdd():
            if other.is_rdd():
                oper2 = other.to_path(min_partitions, True)

                so = ([], [], [])

                with fi.input(files=glob(oper2.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        if value_type(l[2]) != value_type():
                            so[0].append(int(l[0]))
                            so[1].append(int(l[1]))
                            so[2].append(value_type(l[2]))

                b = broadcast(spark_context, so)

                del so

                oper2.destroy()

                def __map(m):
                    base_i, base_j = m[0] * o_shape[0]
                    k = []
                    for i in range(len(b.value[2])):
                        if b.value[2][i] != 0.0:
                            k.append(
                                "{} {} {}".format(base_i + b.value[0][i], 0, m[2] * b.value[2][i]))
                    return "\n".join(k)

                path = get_tmp_path()

                self.data.map(
                    __map
                ).saveAsTextFile(path)

                return State(
                    path,
                    spark_context,
                    mesh,
                    shape,
                    num_particles,
                    log_filename=self.__logger.filename
                )
            else:
                # TODO
                raise NotImplementedError
        else:
            # TODO
            raise NotImplementedError

    def full_measurement(self, min_partitions=8):
        self.__logger.info("Measuring the state of the system...")
        t1 = datetime.now()

        cs = 2

        path = get_tmp_path()

        if self.__mesh.is_1d():
            ndim = 1
            num_p = self.__num_particles
            ind = ndim * num_p
            dims = [self.__mesh.size for p in range(ind)]

            if self.__num_particles == 1:
                dims.append(1)

            size = self.__mesh.size
            cs_size = cs * size
            r = cs_size ** num_p
        elif self.__mesh.is_2d():
            ndim = 2
            num_p = self.__num_particles
            ind = ndim * num_p
            dims = []

            for p in range(0, ind, ndim):
                dims.append(self.__mesh.size[0])
                dims.append(self.__mesh.size[1])

            size_x = self.__mesh.size[0]
            size_y = self.__mesh.size[1]
            cs_size_x = cs * size_x
            cs_size_y = cs * size_y
            r = (cs_size_x * cs_size_y) ** num_p
        else:
            # TODO
            raise NotImplementedError

        if self.is_dense() or self.is_sparse():
            if self.is_sparse():
                s = broadcast(self.__spark_context, self.data.toarray())
            else:
                s = broadcast(self.__spark_context, self.data)

            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m / (cs_size ** (num_p - 1 - p))) % size)
                    if num_p == 1:
                        a.append("0")
                    a.append((abs(s.value[m, 0]) ** 2).real)
                    return " ".join([str(i) for i in a])
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                    a.append((abs(s.value[m, 0]) ** 2).real)
                    return " ".join([str(i) for i in a])
            else:
                # TODO
                raise NotImplementedError

            self.__spark_context.range(
                r, numSlices=min_partitions
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)

            s.unpersist()
        elif self.is_rdd():
            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)
                    if num_p == 1:
                        a.append(m[1])
                    a.append((abs(m[2]) ** 2).real)
                    return a
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                    a.append((abs(m[2]) ** 2).real)
                    return a
            else:
                # TODO
                raise NotImplementedError

            full_measurement = self.data.filter(
                lambda m: m[2] != (0+0j)
            ).map(
                __map
            )

            pdf = PDF(
                full_measurement, self.__spark_context, self.__mesh, dims, log_filename=self.__logger.filename
            ).to_dense()

            self.__logger.info("Checking if the probabilities sum one...")
            if not pdf.sums_one():
                # TODO
                raise ValueError("PDFs must sum one")

            self.__logger.info("Full measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

            return pdf
        elif self.is_path():
            def __smap(m):
                a = m.split()
                return int(a[0]), complex(a[2])

            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)
                    if num_p == 1:
                        a.append("0")
                    a.append((abs(m[1]) ** 2).real)
                    return " ".join([str(i) for i in a])
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                    a.append((abs(m[1]) ** 2).real)
                    return " ".join([str(i) for i in a])
            else:
                # TODO
                raise NotImplementedError

            self.__spark_context.textFile(
                self.data, minPartitions=min_partitions
            ).map(
                __smap
            ).filter(
                lambda m: m[1] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)
        else:
            raise NotImplementedError

        full_measurement = np.zeros(dims, dtype=float)

        ind = len(dims)

        with fi.input(files=glob(path + '/part-*')) as f:
            for line in f:
                l = line.split()
                for i in range(ind):
                    l[i] = int(l[i])
                full_measurement[tuple(l[0:ind])] += float(l[ind])

        remove_tmp_path(path)

        pdf = PDF(full_measurement, self.__spark_context, self.__mesh, log_filename=self.__logger.filename)

        self.__logger.info("Checking if the probabilities sum one...")
        if not pdf.sums_one():
            # TODO
            raise ValueError("PDFs must sum one")

        self.__logger.info("Full measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

        return pdf

    def filtered_measurement(self, full_measurement, min_partitions=8):
        self.__logger.info("Measuring the state of the system which the particles are at the same positions...")
        t1 = datetime.now()

        if not is_pdf(full_measurement):
            raise TypeError("PDF instance expected")

        if self.__mesh.is_1d():
            ndim = 1
            num_p = self.__num_particles
            ind = ndim * num_p
            size = self.__mesh.size
            shape = (size, 1)
        elif self.__mesh.is_2d():
            ndim = 2
            num_p = self.__num_particles
            ind = ndim * num_p
            size_x = self.__mesh.size[0]
            size_y = self.__mesh.size[1]
            shape = (size_x, size_y)
        else:
            # TODO
            raise NotImplementedError

        if full_measurement.is_dense() or full_measurement.is_sparse():
            if self.__mesh.is_1d():
                filtered_measurement = sp.dok_matrix(shape)

                t = [0 for p in range(self.__num_particles)]

                for x in range(size):
                    for p in range(self.__num_particles):
                        t[p] = x

                    if full_measurement.data[tuple(t)] != 0.0:
                        filtered_measurement[x, 0] = full_measurement.data[tuple(t)]
            elif self.__mesh.is_2d():
                filtered_measurement = sp.dok_matrix(shape)

                t = [0 for p in range(ind)]

                for x in range(size_x):
                    for y in range(size_y):
                        for p in range(0, ind, ndim):
                            t[p] = x
                            t[p + 1] = y

                        if full_measurement.data[tuple(t)] != 0.0:
                            filtered_measurement[x, y] = full_measurement.data[tuple(t)]
            else:
                # TODO
                raise NotImplementedError
        elif full_measurement.is_rdd():
            if self.__mesh.is_1d():
                def __filter(m):
                    for p in range(num_p):
                        if m[0] != m[p]:
                            return False
                    return True
            elif self.__mesh.is_2d():
                def __filter(m):
                    for p in range(0, ind, ndim):
                        if m[0] != m[p] or m[1] != m[p + 1]:
                            return False
                    return True
            else:
                # TODO
                raise NotImplementedError

            filtered_measurement = full_measurement.filter(
                __filter
            )

            pdf = PDF(
                filtered_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            ).to_dense()

            self.__logger.info("Filtered measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

            return pdf
        elif full_measurement.is_path():
            if self.__mesh.is_1d():
                def __filter(m):
                    a = m.split()
                    for p in range(num_p):
                        if a[0] != a[p]:
                            return False
                    return True

                def __map(m):
                    a = m.split()
                    return "{} {} {}".format(int(a[0]), 0, float(a[num_p]))
            elif self.__mesh.is_2d():
                def __filter(m):
                    a = m.split()
                    for p in range(0, ind, ndim):
                        if a[0] != a[p] or a[1] != a[p + 1]:
                            return False
                    return True

                def __map(m):
                    a = m.split()
                    return "{} {} {}".format(int(a[0]), int(a[1]), float(a[ind]))
            else:
                # TODO
                raise NotImplementedError

            path = get_tmp_path()

            self.__spark_context.textFile(
                full_measurement, minPartitions=min_partitions
            ).filter(
                __filter
            ).map(
                __map
            ).saveAsTextFile(path)

            filtered_measurement = np.zeros(shape, dtype=float)

            with fi.input(files=glob(path + '/part-*')) as f:
                for line in f:
                    l = line.split()
                    full_measurement[int(l[0]), int(l[1])] += float(l[2])

            remove_tmp_path(path)
        else:
            # TODO
            raise NotImplementedError

        pdf = PDF(filtered_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename)

        self.__logger.info("Filtered measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

        return pdf

    def __partial_measurement(self, particle, min_partitions):
        self.__logger.info("Measuring the state of the system for particle {}...".format(particle + 1))
        t1 = datetime.now()

        cs = 2

        if self.__mesh.is_1d():
            ndim = 1
            num_p = self.__num_particles
            ind = ndim * num_p
            size = self.__mesh.size
            cs_size = cs * size
            shape = (size, 1)
        elif self.__mesh.is_2d():
            ndim = 2
            num_p = self.__num_particles
            ind = ndim * num_p
            size_x = self.__mesh.size[0]
            size_y = self.__mesh.size[1]
            cs_size_x = cs * size_x
            cs_size_y = cs * size_y
            shape = (size_x, size_y)
        else:
            # TODO
            raise NotImplementedError

        if self.is_dense() or self.is_sparse():
            path = get_tmp_path()

            if self.is_dense():
                s = broadcast(self.__spark_context, self.data)
            else:
                s = broadcast(self.__spark_context, self.data.toarray())

            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m / (cs_size ** (num_p - 1 - p2))) % size)
                    return "{} {} {}".format(a[particle], 0, (abs(s.value[m, 0]) ** 2).real)

                r = cs_size ** num_p
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                    return "{} {} {}".format(a[particle], a[particle + 1], (abs(s.value[m, 0]) ** 2).real)

                r = (cs_size_x * cs_size_y) ** num_p
            else:
                # TODO
                raise NotImplementedError

            self.__spark_context.range(
                r, numSlices=min_partitions
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)

            s.unpersist()
        elif self.is_rdd():
            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p2))) % size)
                    return a[particle], m[1], (abs(m[2]) ** 2).real
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                    return a[particle], a[particle + 1], (abs(m[2]) ** 2).real
            else:
                # TODO
                raise NotImplementedError

            partial_measurement = self.data.filter(
                lambda m: m[2] != (0+0j)
            ).map(
                __map
            )

            pdf = PDF(
                partial_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            ).to_dense()

            self.__logger.info("Checking if the probabilities sum one...")
            if not pdf.sums_one():
                # TODO
                raise ValueError("Probabilities must sum one")

            self.__logger.info(
                "Partial measurement for particle {} was done in {}s".format(
                    particle + 1, (datetime.now() - t1).total_seconds()
                )
            )

            return pdf
        elif self.is_path():
            path = get_tmp_path()

            def __smap(m):
                a = m.split()
                return int(a[0]), complex(a[2])

            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p2))) % size)
                    return "{} {} {}".format(a[particle], 0, (abs(m[1]) ** 2).real)
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                    return "{} {} {}".format(a[particle], a[particle + 1], (abs(m[1]) ** 2).real)
            else:
                # TODO
                raise NotImplementedError

            self.__spark_context.range(
                self.data, numSlices=min_partitions
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)
        else:
            # TODO
            raise NotImplementedError

        partial_measurement = np.zeros(shape, dtype=float)

        with fi.input(files=glob(path + '/part-*')) as f:
            for line in f:
                l = line.split()
                partial_measurement[int(l[0]), int(l[1])] += float(l[2])

        remove_tmp_path(path)

        pdf = PDF(
            partial_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
        )

        self.__logger.info("Checking if the probabilities sum one...")
        if not pdf.sums_one():
            # TODO
            raise ValueError("Probabilities must sum one")

        self.__logger.info(
            "Partial measurement for particle {} was done in {}s".format(
                particle + 1, (datetime.now() - t1).total_seconds()
            )
        )

        return pdf

    def partial_measurements(self, particles, min_partitions=8):
        return [self.__partial_measurement(p, min_partitions) for p in particles]


def is_state(obj):
    return isinstance(obj, State)
