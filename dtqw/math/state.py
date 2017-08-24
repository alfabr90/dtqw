import os
import shutil
import fileinput as fi
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from datetime import datetime
from glob import glob
from pyspark import RDD, StorageLevel

from .pdf import PDF, is_pdf
from dtqw.utils.logger import Logger
from dtqw.utils.metrics import Metrics
from dtqw.utils.utils import is_shape, convert_sparse, get_size_of, get_tmp_path, remove_tmp_path, broadcast

__all__ = ['State', 'is_state']


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
                self.__logger.error("Invalid shape")
                raise ValueError("invalid shape")

        if type(arg1) == str:
            self.__from_path(arg1, shape)
        elif isinstance(arg1, RDD):
            self.__from_rdd(arg1, shape, rdd_path)
        elif isinstance(arg1, np.ndarray):
            self.__from_dense(arg1)
        elif sp.isspmatrix(arg1):
            self.__from_sparse(arg1)
        elif isinstance(arg1, (list, tuple)):
            self.__from_block(arg1, shape)
        else:
            self.__logger.error("Invalid argument to instantiate a State object")
            raise TypeError("invalid argument to instantiate a State object")

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

    def __from_block(self, blocks, shape):
        self.data = blocks
        self.shape = shape
        self.__format = 'block'
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

    def persist(self, storage_level=None):
        if storage_level is None:
            storage_level = StorageLevel.MEMORY_AND_DISK

        if self.is_rdd():
            if not self.data.is_cached:
                self.data.persist(storage_level)
                self.__logger.info("RDD {} was persisted".format(self.data.id()))
            else:
                self.__logger.info("RDD {} has already been persisted".format(self.data.id()))
        elif self.is_block():
            for i in range(len(self.data)):
                self.__logger.info("Persisting block {}...".format(i))
                self.data[i].persist(storage_level)
        else:
            self.__logger.warning("It is not possible to persist a non RDD format State")

        return self

    def unpersist(self):
        if self.is_rdd():
            if self.data is not None:
                self.data.unpersist()
                self.__logger.info("RDD {} was unpersisted".format(self.data.id()))
            else:
                self.__logger.info("The RDD has already been unpersisted")
        elif self.is_block():
            if self.data is not None:
                for i in range(len(self.data)):
                    self.__logger.info("Unpersisting block {}...".format(i))
                    self.data[i].unpersist()
        else:
            self.__logger.warning("It is not possible to unpersist a non RDD format State")

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
                for i in range(len(self.data)):
                    self.__logger.info("Destroying block {}...".format(i))
                    self.data[i].destroy()

        self.data = None
        self.__logger.info("State was destroyed")
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
            self.__logger.warning("It is not possible to do a repartition on a non RDD format State")

        return self

    def materialize(self, storage_level=None):
        if storage_level is None:
            storage_level = StorageLevel.MEMORY_AND_DISK

        if self.is_rdd():
            if not self.data.is_cached:
                # self.data = self.data.filter(lambda m: m is not None)
                self.persist(storage_level)
                self.data.count()
                self.__logger.info("State was materialized")
            else:
                self.__logger.info("RDD {} has already been persisted".format(self.data.id()))
        elif self.is_block():
            for i in range(len(self.data)):
                self.__logger.info("Materializing block {}...".format(i))
                self.data[i].materialize(storage_level)

            # self.__logger.info("Operator was materialized")
        else:
            self.__logger.warning("It is not possible to materialize a non RDD format State")

        return self

    def clear_rdd_path(self, storage_level=None):
        if storage_level is None:
            storage_level = StorageLevel.MEMORY_AND_DISK

        if self.is_rdd():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__logger.info("Path was removed")
            self.__rdd_path = None
        elif self.is_block():
            self.materialize(storage_level)
            remove_tmp_path(self.__rdd_path)
            self.__logger.info("Path was removed")
            self.__rdd_path = None

            for i in range(len(self.data)):
                self.__logger.info("Removing path of block {}...".format(i))
                self.data[i].clear_rdd_path(storage_level)
        else:
            self.__logger.warning("It is not possible to clear the path of a non RDD format State")

        return self

    def checkpoint(self):
        if self.is_rdd():
            if not self.data.is_cached:
                self.__logger.warning("It is recommended to cache the RDD before checkpointing")

            self.data.checkpoint()
            self.__logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))
        elif self.is_block():
            for i in range(len(self.data)):
                self.__logger.info("Checkpointing block {}...".format(i))
                self.data[i].checkpoint()

            # self.__logger.info("State was checkpointed")
        else:
            self.__logger.warning("It is not possible to checkpoint a non RDD format State")

        return self

    def is_unitary(self, round_precision=10):
        if self.is_rdd():
            n = self.data.filter(
                lambda m: m[1] != complex()
            ).map(
                lambda m: math.sqrt(m[1].real ** 2 + m[1].imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, round_precision) != 1.0
        elif self.is_block():
            n = 0

            for block in self.data:
                n += block.data.filter(
                    lambda m: m[1] != complex()
                ).map(
                    lambda m: math.sqrt(m[1].real ** 2 + m[1].imag ** 2)
                ).reduce(
                    lambda a, b: a + b
                )

            return round(n, round_precision) != 1.0
        elif self.is_dense():
            return round(np.linalg.norm(self.data), round_precision) == 1.0
        elif self.is_sparse():
            return round(splinalg.norm(self.data), round_precision) == 1.0
        else:
            self.__logger.error("State format not implemented")
            raise NotImplementedError("State format not implemented")

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
        elif self.is_block():
            blocks = []

            for b1 in self.data:
                blocks.append(b1.copy())

            return State(
                blocks,
                self.__spark_context,
                self.__mesh,
                self.shape,
                self.__num_particles,
                log_filename=self.__logger.filename
            )
        else:
            self.__logger.error("Operation not implemented for this State format")
            raise NotImplementedError("operation not implemented for this State format")

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
                    lambda m: "{} {}".format(m[0], m[1])
                ).saveAsTextFile(path)
            elif self.is_dense():
                ind = self.data.nonzero()

                path = get_tmp_path()
                os.mkdir(path)

                files = [open(path + "/part-" + str(i), 'w') for i in range(min_partitions)]

                f = 0

                for i in range(len(ind[0])):
                    files[f].write("{} {}\n".format(ind[0][i], self.data[ind[0][i], ind[1][i]]))
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
                    files[f].write("{} {}\n".format(ind[0][i], ind[2][i]))
                    f = (f + 1) % min_partitions

                for f in files:
                    f.close()

                del ind
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

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

            if self.is_block():
                rdd = self.__spark_context.emptyRDD()

                if len(self.data):
                    block_shape = self.data[0].shape

                    for i in range(len(self.data)):
                        rdd = rdd.union(
                            self.data[i].data.map(
                                lambda m, i=i: (m[0] + i * block_shape[0], m[1])
                            )
                        )

                    rdd = rdd.filter(
                        lambda m: m[1] != value_type()
                    )

                rdd_path = self.__rdd_path
            else:
                oper = self.to_path(min_partitions, copy)

                def __map(m):
                    a = m.split()
                    return int(a[0]), value_type(a[1])

                rdd = self.__spark_context.textFile(
                    oper.data  # , minPartitions=min_partitions
                ).map(
                    __map
                ).filter(
                    lambda m: m[1] != value_type()
                )

                rdd_path = oper.data

            if copy:
                return State(
                    rdd,
                    self.__spark_context,
                    self.__mesh,
                    self.shape,
                    self.__num_particles,
                    rdd_path=rdd_path,
                    log_filename=self.__logger.filename
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
                dense = np.zeros(self.shape, dtype=complex)

                with fi.input(files=glob(self.data + '/part-*')) as f:
                    for line in f:
                        l = line.split()
                        dense[int(l[0]), 0] += complex(l[1])
            elif self.is_rdd():
                # '''
                dense = np.zeros(self.shape, dtype=complex)

                for i in self.data.collect():
                    dense[i[0], 0] += i[1]
            elif self.is_sparse():
                dense = self.data.toarray()
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

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
                self.__memory_usage = self.__get_bytes()
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
                        sparse[int(l[0]), 0] = complex(l[1])

                sparse = convert_sparse(sparse, format)
            elif self.is_rdd():
                # '''
                shape = self.shape

                def __map(m):
                    k = sp.dok_matrix(shape, dtype=complex)
                    k[m[0], 0] = m[1]
                    return k.tocsc()

                sparse = convert_sparse(
                    self.data.filter(
                        lambda m: m[1] != complex()
                    ).map(
                        __map
                    ).reduce(
                        lambda a, b: a + b
                    ),
                    format
                )
            elif self.is_dense():
                sparse = convert_sparse(sp.coo_matrix(self.data), format)
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

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
                self.__memory_usage = self.__get_bytes()
                return self

    def to_block(self, num_blocks, min_partitions=8, storage_level=None, copy=False):
        if self.is_block():
            if copy:
                return self.copy()
            else:
                return self
        else:
            oper = self.to_rdd(min_partitions, copy)

            if self.shape[0] % num_blocks != 0:
                self.__logger.error("Incompatible number of blocks")
                raise ValueError("incompatible number of blocks")

            blocks = []
            block_shape = (int(self.shape[0] / num_blocks), 1)

            num_partitions = oper.data.getNumPartitions()

            for i in range(num_blocks):
                blocks.append(
                    State(
                        oper.data.filter(
                            lambda m, i=i: i * block_shape[0] <= m[0] < (i + 1) * block_shape[0]
                        ).map(
                            lambda m, i=i: (m[0] - i * block_shape[0], m[1])
                        ).partitionBy(numPartitions=num_partitions),
                        self.__spark_context,
                        self.__mesh,
                        block_shape,
                        self.__num_particles,
                        rdd_path=self.__rdd_path,
                        log_filename=self.__logger.filename
                    )
                )

            for i in range(num_blocks):
                t1 = datetime.now()
                self.__logger.debug("Materializing State block {}...".format(i))

                blocks[i].materialize(storage_level)

                self.__logger.debug(
                    "State block {} was materialized in {}s".format(
                        i, (datetime.now() - t1).total_seconds()
                    )
                )

            if copy:
                return State(
                    blocks,
                    self.__spark_context,
                    self.__mesh,
                    self.shape,
                    self.__num_particles,
                    rdd_path=oper.rdd_path,
                    log_filename=self.__logger.filename
                )
            else:
                self.destroy()
                self.data = blocks
                self.__format = 'block'
                self.__memory_usage = self.__get_bytes()
                return self

    def kron(self, other, min_partitions=8):
        if not is_state(other):
            self.__logger.error('State instance expected, not "{}"'.format(type(other)))
            raise TypeError('State instance expected, not "{}"'.format(type(other)))

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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")
        elif self.is_rdd():
            if other.is_rdd():
                oper2 = other.to_path(min_partitions, True)

                so = ([], [])

                with fi.input(files=glob(oper2.data + '/part-*')) as f:
                    for line in f:
                            l = line.split()
                            if value_type(l[2]) != value_type():
                                so[0].append(int(l[0]))
                                so[1].append(value_type(l[1]))

                b = broadcast(spark_context, so)

                del so

                oper2.destroy()

                def __map(m):
                    base_i = m[0] * o_shape[0]
                    k = []
                    for i in range(len(b.value[1])):
                        if b.value[1][i] != value_type():
                            k.append(
                                "{} {}".format(base_i + b.value[0][i], m[1] * b.value[1][i]))
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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")
        else:
            self.__logger.error("Operation not implemented for this State format")
            raise NotImplementedError("operation not implemented for this State format")

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
            self.__logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

            self.__spark_context.range(
                r  # , numSlices=min_partitions
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)

            s.unpersist()

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
        elif self.is_rdd():
            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)
                    if num_p == 1:
                        a.append(0)
                    a.append((abs(m[1]) ** 2).real)
                    return a
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                    a.append((abs(m[1]) ** 2).real)
                    return a
            else:
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

            full_measurement = self.data.filter(
                lambda m: m[1] != (0+0j)
            ).map(
                __map
            )

            pdf = PDF(
                full_measurement, self.__spark_context, self.__mesh, dims, log_filename=self.__logger.filename
            ).to_dense()
        else:
            self.__logger.error("Operation not implemented for this State format")
            raise NotImplementedError("operation not implemented for this State format")

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        self.__logger.info("Checking if the probabilities sum one...")
        if not pdf.sums_one():
            self.__logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        self.__logger.info("Full measurement was done in {}s".format((datetime.now() - t1).total_seconds()))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return pdf

    def filtered_measurement(self, full_measurement):
        self.__logger.info("Measuring the state of the system which the particles are at the same positions...")
        t1 = datetime.now()

        if not is_pdf(full_measurement):
            self.__logger.error('PDF instance expected, not "{}"'.format(type(full_measurement)))
            raise TypeError('PDF instance expected, not "{}"'.format(type(full_measurement)))

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
            self.__logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

            pdf = PDF(
                filtered_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            )
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
                self.__logger.error("Operation not implemented for this State format")
                raise NotImplementedError("operation not implemented for this State format")

            filtered_measurement = full_measurement.filter(
                __filter
            )

            pdf = PDF(
                filtered_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            ).to_dense()
        else:
            self.__logger.error("Operation not implemented for this State format")
            raise NotImplementedError("operation not implemented for this State format")

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
            self.__logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

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
                self.__logger.error("Mesh dimension not implemented")
                raise NotImplementedError("mesh dimension not implemented")

            self.__spark_context.range(
                r  # , numSlices=min_partitions
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)

            s.unpersist()

            partial_measurement = np.zeros(shape, dtype=float)

            with fi.input(files=glob(path + '/part-*')) as f:
                for line in f:
                    l = line.split()
                    partial_measurement[int(l[0]), int(l[1])] += float(l[2])

            remove_tmp_path(path)

            pdf = PDF(
                partial_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            )
        elif self.is_rdd():
            if self.__mesh.is_1d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / (cs_size ** (num_p - 1 - p2))) % size)
                    return a[particle], 0, (abs(m[1]) ** 2).real
            elif self.__mesh.is_2d():
                def __map(m):
                    a = []
                    for p2 in range(num_p):
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                        a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                    return a[particle], a[particle + 1], (abs(m[1]) ** 2).real
            else:
                self.__logger.error("Mesh dimension not implemented")
                raise NotImplementedError("mesh dimension not implemented")

            partial_measurement = self.data.filter(
                lambda m: m[1] != (0+0j)
            ).map(
                __map
            )

            pdf = PDF(
                partial_measurement, self.__spark_context, self.__mesh, shape, log_filename=self.__logger.filename
            ).to_dense()
        else:
            self.__logger.error("Operation not implemented for this State format")
            raise NotImplementedError("operation not implemented for this State format")

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        self.__logger.info("Checking if the probabilities sum one...")
        if not pdf.sums_one():
            self.__logger.error("PDFs must sum one")
            raise ValueError("PDFs must sum one")

        self.__logger.info(
            "Partial measurement for particle {} was done in {}s".format(
                particle + 1, (datetime.now() - t1).total_seconds()
            )
        )

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return pdf

    def partial_measurements(self, particles, min_partitions=8):
        return [self.__partial_measurement(p, min_partitions) for p in particles]


def is_state(obj):
    return isinstance(obj, State)
