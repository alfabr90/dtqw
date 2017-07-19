import os
import shutil
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import fileinput as fi

from glob import glob
from pyspark import RDD, StorageLevel

from .logger import Logger
from .metrics import Metrics
from .state import State, is_state
from .utils import is_shape, convert_sparse, broadcast, get_size_of, get_tmp_path, remove_tmp_path

__all__ = ['Operator', 'is_operator']


class Operator:
    def __init__(self, arg1, spark_context, shape=None, value_type=complex, rdd_path=None, log_filename='log.txt'):
        self.__spark_context = spark_context
        self.__format = None
        self.__value_type = None
        self.__memory_usage = None
        self.__rdd_path = None
        self.__logger = Logger(__name__, log_filename)
        self.__metrics = Metrics(log_filename=log_filename)

        self.data = None
        self.shape = None

        if shape is not None:
            if not is_shape(shape):
                raise ValueError("Not a valid shape")

        if type(arg1) == str:
            self.__from_path(arg1, shape, value_type)
        elif isinstance(arg1, RDD):
            self.__from_rdd(arg1, shape, value_type, rdd_path)
        elif isinstance(arg1, np.ndarray):
            self.__from_dense(arg1)
        elif sp.isspmatrix(arg1):
            self.__from_sparse(arg1)
        else:
            raise TypeError

    @property
    def format(self):
        return self.__format

    @property
    def spark_context(self):
        return self.__spark_context

    @property
    def value_type(self):
        return self.__value_type

    @property
    def memory_usage(self):
        return self.__memory_usage

    @property
    def rdd_path(self):
        return self.__rdd_path

    def __from_path(self, path, shape, value_type):
        self.data = path
        self.shape = shape
        self.__format = 'path'
        self.__value_type = value_type
        self.__memory_usage = self.__get_bytes()

    def __from_rdd(self, rdd, shape, value_type, rdd_path):
        self.data = rdd
        self.shape = shape
        self.__format = 'rdd'
        self.__value_type = value_type
        self.__rdd_path = rdd_path
        self.__memory_usage = self.__get_bytes()

    def __from_dense(self, dense):
        self.data = dense
        self.shape = dense.shape
        self.__format = 'dense'
        self.__value_type = dense.dtype.type
        self.__memory_usage = self.__get_bytes()

    def __from_sparse(self, sparse):
        self.data = sparse
        self.shape = sparse.shape
        self.__format = 'sparse'
        self.__value_type = sparse.dtype.type
        self.__memory_usage = self.__get_bytes()

    def __get_bytes(self):
        return get_size_of(self.data)

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
            return Operator(
                path, self.__spark_context, self.shape, self.__value_type, log_filename=self.__logger.filename
            )
        elif self.is_rdd():
            operator = self.to_path(self.data.getNumPartitions(), True)
            return operator.to_rdd(self.data.getNumPartitions())
        elif self.is_dense():
            return Operator(self.data.copy(), self.__spark_context, log_filename=self.__logger.filename)
        elif self.is_sparse():
            return Operator(self.data.copy(), self.__spark_context, log_filename=self.__logger.filename)

    def repartition(self, num_partitions):
        if self.is_rdd():
            if self.data.getNumPartitions() > num_partitions:
                self.__logger.info(
                    "As this RDD has more partitions than the desired, "
                    "it will be coalesced into {} partitions".format(num_partitions)
                )
                self.data = self.data.coalesce(num_partitions)
            elif self.data.getNumPartitions() < num_partitions:
                self.__logger.info(
                    "As this RDD has less partitions than the desired, "
                    "it will be repartitioned into {} partitions".format(num_partitions)
                )
                self.data = self.data.repartition(num_partitions)
            else:
                self.__logger.info(
                    "As this RDD has many partitions than the desired, there is nothing to do".format(num_partitions)
                )

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            self.data = self.data.map(lambda m: m)
            if not self.data.is_cached:
                self.persist(storage_level)
            self.data.count()

    def clear_rdd_path(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None

    def is_unitary(self, round_precision=10):
        value_type = self.__value_type

        if self.is_path():
            n = self.data.map(
                lambda m: value_type(m.split()[2])
            ).filter(
                lambda m: m != value_type
            ).map(
                lambda m: math.sqrt(m.real ** 2 + m.imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_rdd():
            n = self.data.filter(
                lambda m: m[2] != value_type
            ).map(
                lambda m: math.sqrt(m[2].real ** 2 + m[2].imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_dense():
            return round(splinalg.norm(self.data), round_precision) == 1.0
        elif self.is_sparse():
            return round(np.linalg.norm(self.data), round_precision) == 1.0
        else:
            # TODO
            raise NotImplementedError

    def to_path(self, min_partitions=8, copy=False):
        if self.is_path():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_rdd():
                path = get_tmp_path()

                self.data.map(
                    lambda m: "{} {} {}".format(m[0], m[1], m[2])
                ).saveAsTextFile(path)
            elif self.is_dense():
                ind = self.data.nonzero()

                path = get_tmp_path()
                os.mkdir(path)

                files = [open(path + "/part-" + str(i), 'w') for i in range(min_partitions)]

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

                files = [open(path + "/part-" + str(i), 'w') for i in range(min_partitions)]

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
                return Operator(
                    path, self.__spark_context, self.shape, self.__value_type, log_filename=self.__logger.filename
                )
            else:
                self.destroy()
                self.data = path
                self.__format = 'path'
                self.__memory_usage = self.__get_bytes()
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
                return Operator(
                    rdd, self.__spark_context, self.shape, rdd_path=oper.data, log_filename=self.__logger.filename
                )
            else:
                self.__rdd_path = oper.data
                self.data = rdd
                self.__format = 'rdd'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_dense(self, copy=False):
        if self.is_dense():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_path():
                dense = np.zeros(self.shape, dtype=self.__value_type)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        dense[int(l[0]), int(l[1])] += self.__value_type(l[2])
            elif self.is_rdd():
                dense = np.zeros(self.shape, dtype=self.__value_type)

                for i in self.data.collect():
                    dense[i[0], i[1]] += i[2]
            elif self.is_sparse():
                dense = self.data.toarray()
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return Operator(dense, self.__spark_context, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = dense
                self.__format = 'dense'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_sparse(self, format='csc', copy=False):
        if self.is_sparse():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_path():
                sparse = sp.dok_matrix(self.shape, dtype=self.__value_type)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        sparse[int(l[0]), int(l[1])] = self.__value_type(l[2])
            elif self.is_rdd():
                shape = self.shape
                value_type = self.__value_type

                def __map(m):
                    k = sp.dok_matrix(shape, dtype=value_type)
                    k[m[0], m[1]] = m[2]
                    return k.tocsc()

                sparse = self.data.filter(
                    lambda m: m[2] != value_type()
                ).map(
                    __map
                ).reduce(
                    lambda a, b: a + b
                )
            elif self.is_dense():
                sparse = convert_sparse(sp.coo_matrix(self.data), format)
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return Operator(sparse, self.__spark_context, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = sparse
                self.__format = 'sparse'
                self.__memory_usage = self.__get_bytes()
                return self

    def is_path(self):
        return self.__format == 'path'

    def is_rdd(self):
        return self.__format == 'rdd'

    def is_dense(self):
        return self.__format == 'dense'

    def is_sparse(self):
        return self.__format == 'sparse'

    def __multiply_operator(self, other, min_partitions=8):
        if self.shape[1] != other.shape[0]:
            raise ValueError('incompatible shapes {} and {}'.format(self.shape, other.shape))

        value_type = self.__value_type
        spark_context = self.__spark_context
        shape = (self.shape[0], other.shape[1])

        if self.is_sparse():
            if other.is_sparse():
                oper1 = broadcast(self.__spark_context, self.data.tocsc())
                oper2 = broadcast(other.spark_context, other.data.tocsr())

                c = other.shape[1]

                data = self.__spark_context.range(
                    c, numSlices=min_partitions
                ).map(
                    lambda m: oper1.value[:, m] * oper2.value[m, :]
                ).reduce(
                    lambda a, b: a + b
                )

                oper1.unpersist()
                oper2.unpersist()

                return Operator(data, spark_context, shape, log_filename=self.__logger.filename)
            else:
                # TODO
                raise NotImplementedError
        elif self.is_rdd():
            oper1 = self.data.map(
                lambda m: (m[1], (m[0], m[2]))
            )

            if other.is_rdd():
                oper2 = other.data.map(
                    lambda m: (m[0], (m[1], m[2]))
                )

                j = oper1.join(
                    oper2
                )

                r = j.map(
                    lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
                ).reduceByKey(
                    lambda a, b: a + b, numPartitions=min_partitions
                ).map(
                    lambda m: (m[0][0], m[0][1], m[1])
                )

                j.unpersist()

                return Operator(r, spark_context, shape, log_filename=self.__logger.filename)
            else:
                # TODO
                raise NotImplementedError
        else:
            # TODO
            raise NotImplementedError

    def __multiply_state(self, other, min_partitions=8):
        if self.shape[1] != other.shape[0]:
            raise ValueError('incompatible shapes {} and {}'.format(self.shape, other.shape))

        value_type = complex
        spark_context = self.__spark_context
        shape = other.shape

        if self.is_sparse():
            if other.is_sparse() or other.is_dense():
                oper1 = broadcast(self.__spark_context, self.data.tocsc())

                if other.is_sparse():
                    oper2 = broadcast(other.spark_context, other.data.toarray())
                else:
                    oper2 = broadcast(other.spark_context, other.data)

                data = self.__spark_context.range(
                    shape[0], numSlices=min_partitions
                ).map(
                    lambda m: (m, 0, (oper1.value[m, :] * oper2.value)[0, 0])
                )

                if other.is_sparse():
                    state = State(
                        data, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                    ).to_sparse()
                else:
                    state = State(
                        data, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                    ).to_dense()

                oper1.unpersist()
                oper2.unpersist()
                return state
            else:
                # TODO
                raise NotImplementedError
        elif self.is_rdd():
            oper1 = self.data.map(
                lambda m: (m[1], (m[0], m[2]))
            )

            if other.is_sparse() or other.is_dense():
                if other.is_sparse():
                    oper2 = broadcast(other.spark_context, other.data.toarray())
                else:
                    oper2 = broadcast(other.spark_context, other.data)

                data = oper1.filter(
                    lambda m: oper2.value[m[0], 0] != value_type()
                ).map(
                    lambda m: (m[1][0], 0, m[1][1] * oper2.value[m[0], 0])
                )

                if other.is_sparse():
                    state = State(
                        data, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                    ).to_sparse()
                else:
                    state = State(
                        data, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                    ).to_dense()

                oper2.unpersist()

                return state
            elif other.is_rdd():
                oper2 = other.data.map(
                    lambda m: (m[0], (m[1], m[2]))
                )

                j = oper1.join(
                    oper2
                )

                r = j.map(
                    lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
                ).reduceByKey(
                    lambda a, b: a + b, numPartitions=min_partitions
                ).map(
                    lambda m: (m[0][0], m[0][1], m[1])
                )

                j.unpersist()

                return State(
                    r, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                )
            else:
                # TODO
                raise NotImplementedError
        else:
            # TODO
            raise NotImplementedError

    def multiply(self, other, min_partitions=8):
        if is_operator(other):
            return self.__multiply_operator(other, min_partitions)
        elif is_state(other):
            return self.__multiply_state(other, min_partitions)
        else:
            raise TypeError("State or Operator instance expected")

    def kron(self, other, min_partitions=8):
        if not is_operator(other):
            raise TypeError('Operator instance expected (not "{}")'.format(type(other)))

        value_type = complex
        spark_context = self.__spark_context
        s_shape = self.shape
        o_shape = other.shape
        shape = (self.shape[0] * other.shape[0], self.shape[1] * other.shape[1])

        if self.is_sparse():
            if other.is_sparse():
                return Operator(sp.kron(self.data, other.data), spark_context, log_filename=self.__logger.filename)
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
                    base_i, base_j = m[0] * o_shape[0], m[1] * o_shape[1]
                    k = []
                    for i in range(len(b.value[2])):
                        if b.value[2][i] != 0.0:
                            k.append(
                                "{} {} {}".format(base_i + b.value[0][i], base_j + b.value[1][i], m[2] * b.value[2][i]))
                    return "\n".join(k)

                path = get_tmp_path()

                self.data.map(
                    __map
                ).saveAsTextFile(path)

                return Operator(
                    path, spark_context, shape, log_filename=self.__logger.filename
                ).to_rdd(
                    min_partitions
                )
            else:
                # TODO
                raise NotImplementedError
        else:
            # TODO
            raise NotImplementedError


def is_operator(obj):
    return isinstance(obj, Operator)
