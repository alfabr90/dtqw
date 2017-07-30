import fileinput as fi
import os
import shutil
from glob import glob

import numpy as np
import scipy.sparse as sp
from pyspark import RDD, StorageLevel

from dtqw.utils.logger import Logger
from dtqw.utils.metrics import Metrics
from dtqw.utils.utils import convert_sparse, get_size_of, get_tmp_path, remove_tmp_path

__all__ = ['PDF', 'is_pdf']


class PDF:
    def __init__(self, arg1, spark_context, mesh, shape=None, rdd_path=None, log_filename='log.txt'):
        self.__spark_context = spark_context
        self.__mesh = mesh
        self.__format = None
        self.__value_type = float
        self.__memory_usage = None
        self.__rdd_path = None
        self.__logger = Logger(__name__, log_filename)
        self.__metrics = Metrics(log_filename=log_filename)

        self.data = None
        self.shape = None

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
            return PDF(path, self.__spark_context, self.__mesh, self.shape, log_filename=self.__logger.filename)
        elif self.is_rdd():
            pdf = self.to_path(self.data.getNumPartitions(), True)
            return pdf.to_rdd(self.data.getNumPartitions())
        elif self.is_dense():
            return PDF(self.data.copy(), self.__spark_context, self.__mesh, log_filename=self.__logger.filename)
        elif self.is_sparse():
            return PDF(self.data.copy(), self.__spark_context, self.__mesh, log_filename=self.__logger.filename)
        else:
            # TODO
            raise NotImplementedError

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
            if not self.data.is_cached:
                self.persist(storage_level)
            self.data.count()

    def clear_rdd_path(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None

    def sums_one(self, round_precision=10):
        ind = len(self.shape)

        if self.is_rdd():
            n = self.data.filter(
                lambda m: m[ind] != 0.0
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_dense():
            return round(self.data.sum(), round_precision) == 1.0
        elif self.is_sparse():
            return round(self.data.sum(), round_precision) == 1.0
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
                    lambda m: " ".join(m)
                ).saveAsTextFile(path)
            elif self.is_dense():
                ind = self.data.nonzero()

                path = get_tmp_path()
                os.mkdir(path)

                files = [open(path + "/part-" + str(i), 'w') for i in range(min_partitions)]

                f = 0

                for i in range(len(ind[0])):
                    l = [ind[j][i] for j in range(len(ind))]
                    files[f].write("{} {}\n".format(" ".join(l), self.data[ind[0][i], ind[1][i]]))
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
                return PDF(path, self.__spark_context, self.__mesh, self.shape, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = path
                self.__format = 'path'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_rdd(self, min_partitions=8, copy=False):
        ind = len(self.shape)

        if self.is_rdd():
            if copy:
                return self.copy()
            else:
                return self
        else:
            oper = self.to_path(min_partitions, copy)

            def __map(m):
                a = m.split()
                for i in range(a):
                    a[i] = int(a[i])
                a[ind] = float(a[ind])
                return a

            rdd = self.__spark_context.textFile(
                oper.data, minPartitions=min_partitions
            ).map(
                __map
            ).filter(
                lambda m: m[ind] != float()
            )

            if copy:
                return PDF(
                    rdd,
                    self.__spark_context,
                    self.__mesh,
                    self.shape,
                    rdd_path=oper.data,
                    log_filename=self.__logger.filename
                )
            else:
                self.__rdd_path = oper.data
                self.data = rdd
                self.__format = 'rdd'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_dense(self, copy=False):
        ind = len(self.shape)

        if self.is_dense():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_path():
                dense = np.zeros(self.shape, dtype=float)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        for i in range(ind):
                            l[i] = int(l[i])
                        dense[tuple(l[0:ind])] += float(l[ind])
            elif self.is_rdd():
                dense = np.zeros(self.shape, dtype=float)

                for i in self.data.collect():
                    dense[tuple(i[0:ind])] += i[ind]
            elif self.is_sparse():
                dense = self.data.toarray()
            else:
                # TODO
                raise NotImplementedError

            if copy:
                return PDF(dense, self.__spark_context, self.__mesh, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = dense
                self.__format = 'dense'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_sparse(self, format='csc', copy=False):
        if len(self.shape) > 2:
            raise ValueError('sparse matrices must be of 2 dimensions (not "{}")'.format(len(self.shape)))

        if self.is_sparse():
            if copy:
                return self.copy()
            else:
                return self
        else:
            if self.is_path():
                sparse = sp.dok_matrix(self.shape, dtype=float)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        sparse[int(l[0]), int(l[1])] = float(l[2])
            elif self.is_rdd():
                shape = self.shape

                def __map(m):
                    k = sp.dok_matrix(shape, dtype=float)
                    k[m[0], m[1]] = m[2]
                    return k.tocsc()

                sparse = self.data.filter(
                    lambda m: m[2] != float()
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
                return PDF(sparse, self.__spark_context, self.__mesh, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = sparse
                self.__format = 'sparse'
                self.__memory_usage = self.__get_bytes()
                return self


def is_pdf(obj):
    return isinstance(obj, PDF)
