import os
import shutil
import fileinput as fi
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from glob import glob
from datetime import datetime
from pyspark import RDD, StorageLevel

from .state import State, is_state
from dtqw.utils.logger import Logger
from dtqw.utils.metrics import Metrics
from dtqw.utils.utils import is_shape, convert_sparse, broadcast, get_size_of, get_tmp_path, remove_tmp_path

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
                self.__logger.error("Invalid shape")
                raise ValueError("invalid shape")

        if type(arg1) == str:
            self.__from_path(arg1, shape, value_type)
        elif isinstance(arg1, RDD):
            self.__from_rdd(arg1, shape, value_type, rdd_path)
        elif isinstance(arg1, np.ndarray):
            self.__from_dense(arg1)
        elif sp.isspmatrix(arg1):
            self.__from_sparse(arg1)
        elif isinstance(arg1, (list, tuple)):
            self.__from_block(arg1, shape, value_type)
        else:
            self.__logger.error("Invalid argument to instantiate an Operator object")
            raise TypeError("invalid argument to instantiate an Operator object")

    @property
    def spark_context(self):
        return self.__spark_context

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

    def __from_block(self, blocks, shape, value_type):
        self.data = blocks
        self.shape = shape
        self.__format = 'block'
        self.__value_type = value_type
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

    def is_block(self):
        return self.__format == 'block'

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            if not self.data.is_cached:
                self.data.persist(storage_level)
        elif self.is_block():
            for b1 in self.data:
                for b2 in b1:
                    b2.persist(storage_level)
        else:
            self.__logger.warning("It is not possible to persist a non RDD format Operator")

        return self

    def unpersist(self):
        if self.is_rdd():
            if self.data is not None:
                self.data.unpersist()
        elif self.is_block():
            if self.data is not None:
                for b1 in self.data:
                    for b2 in b1:
                        b2.unpersist()
        else:
            self.__logger.warning("It is not possible to unpersist a non RDD format Operator")

        return self

    def destroy(self):
        self.unpersist()

        if self.is_path():
            remove_tmp_path(self.data)
        elif self.is_rdd():
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None
        elif self.is_block():
            if self.data is not None:
                for b1 in self.data:
                    for b2 in b1:
                        b2.destroy()

        self.data = None
        return self

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
        else:
            self.__logger.warning("It is not possible to do a repartition on a non RDD format Operator")

        return self

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            # self.data = self.data.map(lambda m: m)
            if not self.data.is_cached:
                self.persist(storage_level)
            self.data.count()
        elif self.is_block():
            for b1 in self.data:
                for b2 in b1:
                    # self.__logger.debug("Materializing block...")
                    # self.__logger.debug("Block is cached? {}".format(b2.data.is_cached))
                    b2.materialize(storage_level)
                    # self.__logger.debug("Block is cached? {}".format(b2.data.is_cached))
        else:
            self.__logger.warning("It is not possible to materialize a non RDD format Operator")

        return self

    def clear_rdd_path(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.is_rdd():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None
        elif self.is_block():
            for b1 in self.data:
                for b2 in b1:
                    # self.__logger.debug("Materializing block...")
                    # self.__logger.debug("Block is cached? {}".format(b2.data.is_cached))
                    b2.materialize(storage_level)
                    # self.__logger.debug("Block is cached? {}".format(b2.data.is_cached))

            remove_tmp_path(self.__rdd_path)
            self.__rdd_path = None

            for b1 in self.data:
                for b2 in b1:
                    # self.__logger.debug("Clearing RDD path: {}".format(b2.rdd_path))
                    b2.clear_rdd_path(storage_level)
        else:
            self.__logger.warning("It is not possible to clear the path of a non RDD format Operator")

        return self

    def is_unitary(self, round_precision=10):
        value_type = self.__value_type

        if self.is_rdd():
            n = self.data.filter(
                lambda m: m[2] != value_type
            ).map(
                lambda m: math.sqrt(m[2].real ** 2 + m[2].imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_block():
            n = 0

            for block in self.data:
                n += block.data.filter(
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
            self.__logger.error("Operation not implemented for this Operator format")
            raise NotImplementedError("operation not implemented for this Operator format")

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
        elif self.is_block():
            blocks = []

            for b1 in self.data:
                blk = []
                for b2 in b1:
                    blk.append(b2.copy())
                blocks.append(blk)

            return Operator(
                blocks, self.__spark_context, self.shape, self.__value_type, log_filename=self.__logger.filename
            )
        else:
            self.__logger.error("Operation not implemented for this Operator format")
            raise NotImplementedError("operation not implemented for this Operator format")

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
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")

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

            if self.is_block():
                rdd = self.__spark_context.emptyRDD()

                if len(self.data):
                    if len(self.data[0]):
                        block_shape = self.data[0][0].shape

                        for i in range(len(self.data)):
                            for j in range(len(self.data)):
                                rdd = rdd.union(
                                    self.data[i][j].data.map(
                                        lambda m, i=i, j=j: (m[0] + i * block_shape[0], m[1] + j * block_shape[1], m[2])
                                    )
                                )

                        rdd = rdd.filter(
                            lambda m: m[2] != value_type()
                        )

                rdd_path = self.__rdd_path
            else:
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

                rdd_path = oper.data

            if copy:
                return Operator(
                    rdd, self.__spark_context, self.shape, rdd_path=rdd_path, log_filename=self.__logger.filename
                )
            else:
                self.__rdd_path = rdd_path
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
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")

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
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")

            if copy:
                return Operator(sparse, self.__spark_context, log_filename=self.__logger.filename)
            else:
                self.destroy()
                self.data = sparse
                self.__format = 'sparse'
                self.__memory_usage = self.__get_bytes()
                return self

    def to_block(self, num_blocks, min_partitions=8, copy=False):
        if self.is_block():
            if copy:
                return self.copy()
            else:
                return self
        else:
            # self.__logger.debug("Self RDD path: {}".format(self.__rdd_path))
            oper = self.to_rdd(min_partitions, copy)
            # self.__logger.debug("Oper RDD path: {}".format(oper.rdd_path))
            if self.shape[0] % num_blocks != 0 or self.shape[1] % num_blocks != 0:
                self.__logger.error("Incompatible number of blocks")
                raise ValueError("incompatible number of blocks")

            blocks = []
            block_shape = (int(self.shape[0] / num_blocks), int(self.shape[1] / num_blocks))

            for i in range(num_blocks):
                blk = []
                for j in range(num_blocks):
                    blk.append(
                        Operator(
                            oper.data.filter(
                                lambda m, i=i, j=j: i * block_shape[0] <= m[0] < (i + 1) * block_shape[0] and
                                          j * block_shape[1] <= m[1] < (j + 1) * block_shape[1]
                            ).map(
                                lambda m, i=i, j=j: (m[0] - i * block_shape[0], m[1] - j * block_shape[1], m[2])
                            ),
                            self.__spark_context,
                            block_shape,
                            rdd_path=oper.rdd_path,
                            log_filename=self.__logger.filename
                        )
                    )
                blocks.append(blk)

            if copy:
                return Operator(
                    blocks,
                    self.__spark_context,
                    self.shape,
                    rdd_path=oper.rdd_path,
                    log_filename=self.__logger.filename
                )
            else:
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        t1 = datetime.now()
                        self.__logger.debug("Materializing Operator block {},{}...".format(i, j))

                        blocks[i][j].materialize()

                        self.__logger.debug(
                            "Operator block {},{} was materialized in {}s".format(
                                i, j, (datetime.now() - t1).total_seconds()
                            )
                        )

                self.unpersist()
                self.data = blocks
                self.__format = 'block'
                self.__memory_usage = self.__get_bytes()
                return self

    def __multiply_operator(self, other, min_partitions=8):
        if self.shape[1] != other.shape[0]:
            self.__logger.error("Incompatible shapes {} and {}".format(self.shape, other.shape))
            raise ValueError('incompatible shapes {} and {}'.format(self.shape, other.shape))

        value_type = self.__value_type
        spark_context = self.__spark_context
        shape = (self.shape[0], other.shape[1])
        log_filename = self.__logger.filename

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

                return Operator(data, spark_context, shape, log_filename=log_filename)
            else:
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")
        elif self.is_rdd():
            oper1 = self.data.map(
                lambda m: (m[1], (m[0], m[2]))
            )

            if other.is_rdd():
                oper2 = other.data.map(
                    lambda m: (m[0], (m[1], m[2]))
                )

                r = oper1.join(
                    oper2
                ).map(
                    lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
                ).reduceByKey(
                    lambda a, b: a + b, numPartitions=min_partitions
                ).filter(
                    lambda m: m[1] != value_type()
                ).map(
                    lambda m: (m[0][0], m[0][1], m[1])
                )

                return Operator(r, spark_context, shape, log_filename=log_filename)
            else:
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")
        elif self.is_block():
            if len(self.data) == 0:
                self.__logger.error("No blocks in Operator")
                raise ValueError("no blocks in Operator")

            s_blocks = len(self.data[0])

            if s_blocks == 0:
                self.__logger.error("No blocks in Operator")
                raise ValueError("no blocks in Operator")

            if other.is_block():
                if len(other.data) == 0:
                    self.__logger.error("None blocks in State")
                    raise ValueError("no blocks in State")

                o_blocks = len(other.data)

                if s_blocks != o_blocks:
                    self.__logger.error("Incompatible number of blocks {} and {}".format(s_blocks, o_blocks))
                    raise ValueError("incompatible number of blocks {} and {}".format(s_blocks, o_blocks))

                blocks = []

                block_shape = self.data[0][0].shape

                for i in range(len(self.data)):
                    tmp_blocks = []
                    for j in range(len(self.data)):
                        tmp_blocks2 = []

                        t1 = datetime.now()
                        self.__logger.debug("Building Operator block {},{}...".format(i, j))

                        for k in range(len(self.data)):
                            oper1 = self.data[i][k].data.map(
                                lambda m: (m[1], (m[0], m[2]))
                            )

                            oper2 = other.data[k][j].data.map(
                                lambda m: (m[0], (m[1], m[2]))
                            )

                            r = oper1.join(
                                oper2
                            ).map(
                                lambda m, i=i, j=j: (
                                    (m[1][0][0] + i * block_shape[0], m[1][1][0] + j * block_shape[1]),
                                    m[1][0][1] * m[1][1][1]
                                )
                            )

                            tmp_blocks2.append(
                                Operator(
                                    r, spark_context, block_shape, log_filename=log_filename
                                ).materialize()
                            )

                        rdd = spark_context.emptyRDD()

                        for b in tmp_blocks2:
                            rdd = rdd.union(b.data)
                        # print(rdd.collect())
                        rdd = rdd.reduceByKey(
                            lambda a, b: a + b, numPartitions=min_partitions
                        ).filter(
                            lambda m: m[1] != value_type()
                        ).map(
                            lambda m: (m[0][0] - i * block_shape[0], m[0][1] - j * block_shape[1], m[1])
                        )
                        # print(rdd.collect())
                        tmp_blocks.append(
                            Operator(
                                rdd, spark_context, block_shape, log_filename=log_filename
                            ).to_path(min_partitions).to_rdd(min_partitions)
                        )
                        # print(Operator(
                        #     rdd, spark_context, shape, log_filename=log_filename
                        # ).to_path(min_partitions).to_dense().data)
                        for b in tmp_blocks2:
                            b.destroy()

                        self.__logger.debug(
                            "Operator block {},{} was built in {}s".format(i, j, (datetime.now() - t1).total_seconds())
                        )

                    blocks.append(tmp_blocks)

                return Operator(blocks, spark_context, shape, log_filename=log_filename)
            else:
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")
        else:
            self.__logger.error("Operation not implemented for this Operator format")
            raise NotImplementedError("operation not implemented for this Operator format")

    def __multiply_state(self, other, min_partitions=8):
        if self.shape[1] != other.shape[0]:
            self.__logger.error("Incompatible shapes {} and {}".format(self.shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self.shape, other.shape))

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
                    shape[0]  # , numSlices=min_partitions
                ).map(
                    lambda m: (m, (oper1.value[m, :] * oper2.value)[0, 0])
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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")
        elif self.is_rdd():
            oper1 = self.data.map(
                lambda m: (m[1], (m[0], m[2]))
            )

            if other.is_sparse() or other.is_dense():
                if other.is_sparse():
                    oper2 = broadcast(other.spark_context, other.data.toarray())
                else:
                    oper2 = broadcast(other.spark_context, other.data)
                # print(oper2.value)
                data = oper1.filter(
                    lambda m: oper2.value[m[0], 0] != value_type()
                ).map(
                    lambda m: (m[1][0], m[1][1] * oper2.value[m[0], 0])
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
                oper2 = other.data

                r = oper1.join(
                    oper2
                ).map(
                    lambda m: (m[1][0][0], m[1][0][1] * m[1][1])
                ).filter(
                    lambda m: m[1] != value_type()
                ).reduceByKey(
                    lambda a, b: a + b  # , numPartitions=min_partitions
                )

                return State(
                    r, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                )
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")
        elif self.is_block():
            if len(self.data) == 0:
                self.__logger.error("No blocks in Operator")
                raise ValueError("no blocks in Operator")

            s_blocks = len(self.data[0])

            if s_blocks == 0:
                self.__logger.error("No blocks in Operator")
                raise ValueError("no blocks in Operator")

            if other.is_block():
                if len(other.data) == 0:
                    self.__logger.error("None blocks in State")
                    raise ValueError("no blocks in State")

                o_blocks = len(other.data)

                if s_blocks != o_blocks:
                    self.__logger.error("Incompatible number of blocks {} and {}".format(s_blocks, o_blocks))
                    raise ValueError("incompatible number of blocks {} and {}".format(s_blocks, o_blocks))

                blocks = []

                block_shape = self.data[0][0].shape

                for i in range(len(self.data)):
                    tmp_blocks = []
                    oper2 = broadcast(other.spark_context, other.data[i].to_dense(True).data)
                    # print("result:", i)
                    # print(other.data[i].to_dense(copy=True).data)
                    for j in range(len(self.data)):
                        # print("operator:", j, ",", i)
                        # print(self.data[j][i].to_sparse(copy=True).data)
                        data = self.data[j][i].data.filter(
                            lambda m: oper2.value[m[1], 0] != value_type()
                        ).map(
                            lambda m: (m[0] + j * block_shape[0], m[2] * oper2.value[m[1], 0])
                        )

                        tmp_blocks.append(
                            State(
                                data,
                                spark_context,
                                other.mesh,
                                block_shape,
                                other.num_particles,
                                log_filename=self.__logger.filename
                            ).materialize()
                        )

                    oper2.unpersist()

                    blocks.append(tmp_blocks)

                rdd = spark_context.emptyRDD()

                for i in blocks:
                    for j in i:
                        rdd = rdd.union(j.data)
                # print(rdd.collect())
                rdd = rdd.reduceByKey(
                    lambda a, b: a + b  # , numPartitions=min_partitions
                )
                # print(rdd.collect())
                state = State(
                    rdd, spark_context, other.mesh, shape, other.num_particles, log_filename=self.__logger.filename
                ).to_path(min_partitions)

                for i in blocks:
                    for j in i:
                        j.destroy()

                return state.to_block(o_blocks, min_partitions)
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")
        else:
            self.__logger.error("Operation not implemented for this Operator format")
            raise NotImplementedError("operation not implemented for this Operator format")

    def multiply(self, other, min_partitions=8):
        if is_operator(other):
            return self.__multiply_operator(other, min_partitions)
        elif is_state(other):
            return self.__multiply_state(other, min_partitions)
        else:
            self.__logger.error('State or Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('State or Operator instance expected, not "{}"'.format(type(other)))

    def kron(self, other, min_partitions=8):
        if not is_operator(other):
            self.__logger.error('Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(other)))

        value_type = complex
        spark_context = self.__spark_context
        s_shape = self.shape
        o_shape = other.shape
        shape = (self.shape[0] * other.shape[0], self.shape[1] * other.shape[1])

        if self.is_sparse():
            if other.is_sparse():
                return Operator(sp.kron(self.data, other.data), spark_context, log_filename=self.__logger.filename)
            else:
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")
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
                        if b.value[2][i] != value_type():
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
                self.__logger.error("Operation not implemented for this Operator format")
                raise NotImplementedError("operation not implemented for this Operator format")
        else:
            self.__logger.error("Operation not implemented for this Operator format")
            raise NotImplementedError("operation not implemented for this Operator format")


def is_operator(obj):
    return isinstance(obj, Operator)
