import io
import os
import time
from sys import getsizeof
from functools import partial
import tempfile as tf
import fileinput as fi
from glob import glob
import operator as op
import math
import cmath
from datetime import datetime
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.sparse.linalg import norm
from pyspark import *

STR_FORMAT_INT = 7
STR_FORMAT_FLOAT = 23

SAVE_MODE_MEMORY = 2 ** 0
SAVE_MODE_DISK = 2 ** 1
# SAVE_MODE_RDD = 2 ** 2

ROUND_PRECISION = 10

DEBUG_MODE = True

hadamard_1d = np.array(
    [[1, 1],
     [1, -1]], dtype=complex
) / math.sqrt(2)

hadamard_2d = np.array(
    [[1, 1, 1, 1],
     [1, -1, 1, -1],
     [1, 1, -1, -1],
     [1, -1, -1, 1]], dtype=complex
) / 2.0

grover_2d = np.array(
    [[-1, 1, 1, 1],
     [1, -1, 1, 1],
     [1, 1, -1, 1],
     [1, 1, 1, -1]], dtype=complex
) / 2.0

fourier_2d = np.array(
    [[1, 1, 1, 1],
     [1, 1.0j, -1, -1.0j],
     [1, -1, 1, -1],
     [1, -1.0j, -1, 1.0j]], dtype=complex
) / 2.0


def qubit(x, size=2):
    if x < 0 or x >= size:
        raise Exception("Invalid index for qubit!")

    result = sp.dok_matrix((size, 1))
    result[x, 0] = 1.0

    return result.tocsc()


def space(x, size):
    return qubit(get_pos(x, size), size)


def get_pos(x, size):
    return int(math.ceil((size - 1) / 2.0)) + x


def coin_space(size):
    return sp.identity(size, format='csc')


def bra(a):
    return a.conj().transpose()


def braket(a, b):
    c = bra(a) * b

    if c.shape[0] == 1 and c.shape[1] == 1:
        return c[0, 0]
    else:
        return c


def outer(a, b):
    return a * bra(b)


def get_buffer_size(data_size):
    if io.DEFAULT_BUFFER_SIZE:
        return io.DEFAULT_BUFFER_SIZE * (1 + int(math.sqrt(math.sqrt(data_size))))
    else:
        return 8192 * (1 + int(math.sqrt(math.sqrt(data_size))))


def get_size_of(var):
    if isinstance(var, (list, tuple)):
        size = getsizeof(var)

        if len(var) > 0:
            for i in var:
                size += get_size_of(i)

        return size
    elif isinstance(var, dict):
        size = getsizeof(var)
        
        for k, v in var.items():
            size += get_size_of(v)
        
        return size
    elif isinstance(var, np.ndarray):
        return var.nbytes
    elif sp.isspmatrix_dok(var):
        return getsizeof(var)
    elif sp.isspmatrix_coo(var):
        return var.data.nbytes + var.row.nbytes + var.col.nbytes
    elif sp.isspmatrix_csc(var):
        return var.data.nbytes + var.indptr.nbytes + var.indices.nbytes
    elif sp.isspmatrix_csr(var):
        return var.data.nbytes + var.indptr.nbytes + var.indices.nbytes
    else:
        return getsizeof(var)


def create_dir(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise Exception("Invalid path!")
    else:
        os.makedirs(path)


def get_tmp_path(dir=None):
    if dir is None:
        if not (os.environ.get('SCRATCH') is None):
            d = os.environ['SCRATCH'] + "/"
        else:
            d = "./" 
    else:
        d = dir
    
    tmp_file = tf.NamedTemporaryFile(dir=d)
    tmp_file.close()

    return tmp_file.name


def remove_tmp_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            for i in os.listdir(path):
                os.remove(path + "/" + i)
            os.rmdir(path)
        else:
            os.remove(path)


def size_of_tmp_path(path):
    if os.path.isdir(path):
        size = 0
        for i in os.listdir(path):
            size += size_of_tmp_path(path + "/" + i)
        return size
    else:
        return os.stat(path).st_size


def broadcast(sc, data):
    return sc.broadcast(data)


def isunitary(value, sc=None):
    if sp.isspmatrix(value):
        return round(sp.linalg.norm(value), ROUND_PRECISION) == 1.0
    elif isinstance(value, np.ndarray):
        return round(np.linalg.norm(value), ROUND_PRECISION) == 1.0
    elif isinstance(value, RDD):
        n = value.filter(
            lambda m: m[2] != (0+0j)
        ).map(
            lambda m: math.sqrt(m[2].real ** 2 + m[2].imag ** 2)
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, ROUND_PRECISION) != 1.0
    elif type(value) == str:
        if os.path.exists(value):
            n = sc.textFile(value).map(
                lambda m: complex(m.split()[2])
            ).filter(
                lambda m: m != (0+0j)
            ).map(
                lambda m: math.sqrt(m.real ** 2 + m.imag ** 2)
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, ROUND_PRECISION) != 1.0
        else:
            raise FileNotFoundError
    else:
        raise Exception("Unsupported type!")


def check_probabilities(value, sc=None):
    if sp.isspmatrix(value) or isinstance(value, np.ndarray):
        return round(value.sum(), ROUND_PRECISION) == 1.0
    elif isinstance(value, RDD):
        n = value.map(
            lambda m: float(m[2])
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, ROUND_PRECISION) == 1.0
    elif type(value) == str:
        if os.path.exists(value):
            n = sc.textFile(value).map(
                lambda m: float(m.split()[2])
            ).reduce(
                lambda a, b: a + b
            )

            return round(n, ROUND_PRECISION) == 1.0
        else:
            raise FileNotFoundError
    else:
        raise Exception("Unsupported type!")


def convert_sparse(result, format):
    if format == 'csc':
        return result.tocsc()
    if format == 'csr':
        return result.tocsr()
    if format == 'lil':
        return result.tolil()
    if format == 'dia':
        return result.todia()
    if format == 'coo':
        return result.tocoo()
    if format == 'bsr':
        return result.tobsr()
    if format == 'dense':
        return result.todense()

    return result


def rdd_to_sparse(rdd, shape, value_type=complex, format=None):
    def __map(m):
        k = sp.dok_matrix(shape, dtype=value_type)
        k[m[0], m[1]] = m[2]
        return k.tocsc()

    result = rdd.filter(
        lambda m: m[2] != value_type()
    ).map(
        __map
    ).reduce(
        lambda a, b: a + b
    )

    return convert_sparse(result, format)


def rdd_to_dense(rdd, shape, value_type=complex):
    result = np.zeros(shape, dtype=value_type)

    ind = rdd.collect()

    for i in ind:
        result[i[0], i[1]] += i[2]

    return result


def sparse_to_disk(var, value_type=complex, min_partitions=8, dir=None):
    if sp.isspmatrix(var):
        ind = sp.find(var)

        path = get_tmp_path(dir=dir)
        os.mkdir(path)

        data_size = len(ind) * len(ind[2]) * get_size_of(value_type())
        buffer_size = get_buffer_size(data_size)

        files = [open(path + "/part-" + str(i), 'w', buffer_size) for i in range(min_partitions)]

        f = 0

        for i in range(len(ind[2])):
            files[f].write("{} {} {}\n".format(ind[0][i], ind[1][i], ind[2][i]))
            f = (f + 1) % min_partitions

        for f in files:
            f.close()

        del ind

        return path
    else:
        raise Exception("Unsupported type!")


def dense_to_disk(var, value_type=complex, min_partitions=8, dir=None):
    if isinstance(var, np.ndarray):
        ind = var.nonzero()

        path = get_tmp_path(dir=dir)
        os.mkdir(path)

        data_size = len(ind) * len(ind[0]) * get_size_of(value_type())
        buffer_size = get_buffer_size(data_size)

        files = [open(path + "/part-" + str(i), 'w', buffer_size) for i in range(min_partitions)]

        f = 0

        for i in range(len(ind[0])):
            files[f].write("{} {} {}\n".format(ind[0][i], ind[1][i], var[ind[0][i], ind[1][i]]))
            f = (f + 1) % min_partitions

        for f in files:
            f.close()

        del ind

        return path
    else:
        raise Exception("Unsupported type!")


def disk_to_sparse(path, shape, value_type=complex, format=None):
    result = sp.dok_matrix(shape, dtype=value_type)

    with fi.input(files=glob(path + '/part-*')) as f:
        for line in f:
            l = line.split()
            result[int(l[0]), int(l[1])] = value_type(l[2])

    return convert_sparse(result, format)


def disk_to_dense(path, shape, value_type=complex):
    result = np.zeros(shape, dtype=value_type)

    ind = len(shape)

    with fi.input(files=glob(path + '/part-*')) as f:
        for line in f:
            l = line.split()
            for i in range(ind):
                l[i] = int(l[i])
            result[tuple(l[0:ind])] += value_type(l[ind])

    return result


def build_initial_state(num_dimensions, amplitudes, position, size,
                        entangled=False, coin_space_indices=None, operator=op.add,
                        num_particles=1):
    if num_particles < 1:
        raise Exception("There must be at least one particle!")

    cs = coin_space(2)

    if num_dimensions == 1:
        s = space(position, size)

        if entangled:
            if num_particles == 2:
                state = operator(
                            sp.kron(
                                sp.kron(amplitudes[0][0] * cs[:, coin_space_indices[0][0]], s),
                                sp.kron(amplitudes[1][0] * cs[:, coin_space_indices[1][0]], s),
                                format='csc'
                            ),
                            sp.kron(
                                sp.kron(amplitudes[0][1] * cs[:, coin_space_indices[0][1]], s),
                                sp.kron(amplitudes[1][1] * cs[:, coin_space_indices[1][1]], s),
                                format='csc'
                            )
                        )
            else:
                raise NotImplementedError
        else:
            state = sp.kron(
                amplitudes[0][0] * cs[:, 0] + amplitudes[0][1] * cs[:, 1],
                s,
                format='csc'
            )

            for i in range(1, num_particles, 1):
                state = sp.kron(
                    state,
                    sp.kron(
                        amplitudes[i][0] * cs[:, 0] + amplitudes[i][1] * cs[:, 1],
                        s,
                    ),
                    format='csc'
                )
    elif num_dimensions == 2:
        sx, sy = space(position[0], size[0]), space(position[1], size[1])

        state = sp.kron(
            amplitudes[0][0] * sp.kron(cs[:, 0], cs[:, 0]) +
            amplitudes[0][1] * sp.kron(cs[:, 0], cs[:, 1]) +
            amplitudes[0][2] * sp.kron(cs[:, 1], cs[:, 0]) +
            amplitudes[0][3] * sp.kron(cs[:, 1], cs[:, 1]),
            sp.kron(sx, sy),
            format='csc'
        )

        for i in range(1, num_particles, 1):
            state = sp.kron(
                state,
                sp.kron(
                    amplitudes[i][0] * sp.kron(cs[:, 0], cs[:, 0]) +
                    amplitudes[i][1] * sp.kron(cs[:, 0], cs[:, 1]) +
                    amplitudes[i][2] * sp.kron(cs[:, 1], cs[:, 0]) +
                    amplitudes[i][3] * sp.kron(cs[:, 1], cs[:, 1]),
                    sp.kron(sx, sy),
                    format='csc'
                ),
                format='csc'
            )
    else:
        raise NotImplementedError

    return state


def parallel_kron(operand1, operand2, sc, shape, value_type=complex, min_partitions=8):
    unpersist = True

    if sp.isspmatrix(operand1):
        path = sparse_to_disk(operand1, min_partitions=min_partitions)
        remove_path = True
    elif isinstance(operand1, np.ndarray):
        path = dense_to_disk(operand1, min_partitions=min_partitions)
        remove_path = True
    elif type(operand1) == str:
        path = operand1
        remove_path = False
    else:
        raise Exception("Unsupported type for first operand!")

    if sp.isspmatrix(operand2):
        b = broadcast(sc, sp.find(operand2))
    elif isinstance(operand2, Broadcast):
        b = operand2
        unpersist = False
    elif type(operand2) == str:
        so = ([], [], [])

        with fi.input(files=glob(operand2 + '/part-*')) as f:
            for line in f:
                l = line.split()
                if complex(l[2]) != (0+0j):
                    so[0].append(int(l[0]))
                    so[1].append(int(l[1]))
                    so[2].append(value_type(l[2]))

        b = broadcast(sc, so)

        del so
    else:
        raise Exception("Unsupported type for second operand!")

    def __map(m):
        a = m.split()
        a[0], a[1], a[2] = int(a[0]), int(a[1]), value_type(a[2])

        base_i, base_j = a[0] * shape[0], a[1] * shape[1]
        k = []

        for i in range(len(b.value[2])):
            if b.value[2][i] != 0.0:
                k.append("{} {} {}".format(base_i + b.value[0][i], base_j + b.value[1][i], a[2] * b.value[2][i]))

        return "\n".join(k)

    path2 = get_tmp_path()

    sc.textFile(
        path, minPartitions=min_partitions
    ).map(
        __map
    ).saveAsTextFile(path2)

    if unpersist:
        b.unpersist()

    if remove_path:
        remove_tmp_path(path)

    result = path2

    return result


def mat_vec_product(mat, vec, sc=None, value_type=complex, min_partitions=8, save_mode=SAVE_MODE_MEMORY):
    if sp.isspmatrix(mat) or isinstance(mat, np.ndarray):
        if sp.isspmatrix(vec) or isinstance(vec, np.ndarray):
            result = mat * vec

            if save_mode == SAVE_MODE_DISK:
                if sp.isspmatrix(vec):
                    return sparse_to_disk(result, value_type, min_partitions)
                if isinstance(vec, np.ndarray):
                    return dense_to_disk(result, value_type, min_partitions)

            return result
        else:
            raise Exception("Unsupported type for vector!")
    elif type(mat) == str or isinstance(mat, RDD):
        if type(mat) == str:
            def __map(m):
                a = m.split()
                return int(a[1]), (int(a[0]), value_type(a[2]))

            m_rdd = sc.textFile(
                mat, minPartitions=min_partitions
            ).map(
                __map
            )
        else:
            m_rdd = mat.map(
                lambda m: (m[1], (m[0], m[2]))
            )

        if sp.isspmatrix(vec) or isinstance(vec, np.ndarray) or isinstance(vec, Broadcast):
            if sp.isspmatrix(vec) or isinstance(vec, np.ndarray):
                v = broadcast(sc, vec)
                unpersist = True
            else:
                v = vec
                unpersist = False

            m_rdd = m_rdd.filter(
                lambda m: v.value[m[0], 0] != value_type()
            )

            if save_mode == SAVE_MODE_DISK:
                path = get_tmp_path()

                m_rdd.map(
                    lambda m: "{} {} {}".format(m[1][0], 0, m[1][1] * v.value[m[0], 0])
                ).saveAsTextFile(path)

                result = path
            else:
                m_rdd = m_rdd.map(
                    lambda m: (m[1][0], 0, m[1][1] * v.value[m[0], 0])
                )

                if sp.isspmatrix(v.value):
                    result = rdd_to_sparse(m_rdd, v.value.shape, value_type, v.value.format)
                if isinstance(v.value, np.ndarray):
                    result = rdd_to_dense(m_rdd, v.value.shape, value_type)

            if unpersist:
                v.unpersist()

            return result
        elif type(vec) == str or isinstance(vec, RDD):
            if type(vec) == str:
                def __map(m):
                    a = m.split()
                    return int(a[0]), (int(a[1]), value_type(a[2]))

                v_rdd = sc.textFile(
                    vec, minPartitions=min_partitions
                ).map(
                    __map
                )
            else:
                v_rdd = vec.map(
                    lambda m: (m[0], (m[1], m[2]))
                )

            v_rdd = v_rdd.filter(
                lambda m: m[1][1] != value_type()
            )

            mv_rdd = m_rdd.join(
                v_rdd
            ).map(
                lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
            ).reduceByKey(
                lambda a, b: a + b, numPartitions=min_partitions
            )

            if save_mode == SAVE_MODE_DISK:
                path = get_tmp_path()

                mv_rdd.map(
                    lambda m: "{} {} {}".format(m[0][0], m[0][1], m[1])
                ).saveAsTextFile(path)

                result = path
            else:
                result = mv_rdd.map(
                    lambda m: (m[0][0], m[0][1], m[1])
                )

            return result
        else:
            raise Exception("Unsupported type for vector!")
    else:
        raise Exception("Unsupported type for matrix!")


def mat_mat_product(mat1, mat2, sc=None, value_type=complex, min_partitions=8, save_mode=SAVE_MODE_MEMORY):
    if sp.isspmatrix(mat1) or isinstance(mat1, Broadcast):
        if sp.isspmatrix(mat1):
            m1 = broadcast(sc, mat1)
            shape = mat1.shape[1]
        elif isinstance(mat1, Broadcast):
            m1 = mat1
            shape = m1.value.shape[1]

        if sp.isspmatrix(mat2):
            m2 = broadcast(sc, mat2)
        elif isinstance(mat2, Broadcast):
            m2 = mat2
        else:
            raise Exception("Unsupported type for second matrix!")

        result = sc.range(
            shape, numSlices=min_partitions
        ).map(
            lambda m: m1.value[:, m] * m2.value[m, :]
        ).reduce(
            lambda a, b: a + b
        )

        if sp.isspmatrix(mat1):
            m1.unpersist()

        if sp.isspmatrix(mat2):
            m2.unpersist()

        if save_mode == SAVE_MODE_DISK:
            return sparse_to_disk(result, value_type, min_partitions)

        return result
    elif type(mat1) == str or isinstance(mat1, RDD):
        if type(mat1) == str:
            def __map(m):
                a = m.split()
                return int(a[1]), (int(a[0]), value_type(a[2]))

            mat1_rdd = sc.textFile(
                mat1, minPartitions=min_partitions
            ).map(
                __map
            )
        else:
            mat1_rdd = mat1.map(
                lambda m: (m[1], (m[0], m[2]))
            )

        if sp.isspmatrix(mat2) or type(mat2) == str:
            if sp.isspmatrix(mat2):
                path = sparse_to_disk(mat2, complex, min_partitions)
            else:
                path = mat2

            def __map(m):
                a = m.split()
                return int(a[0]), (int(a[1]), value_type(a[2]))

            mat2_rdd = sc.textFile(
                path, minPartitions=min_partitions
            ).map(
                __map
            )
        elif isinstance(mat2, RDD):
            mat2_rdd = mat2.map(
                lambda m: (m[1], (m[0], m[2]))
            )
        else:
            raise Exception("Unsupported type for second matrix!")

        rdd = mat1_rdd.join(
            mat2_rdd
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=min_partitions
        )

        if save_mode == SAVE_MODE_DISK:
            path = get_tmp_path()

            rdd.map(
                lambda m: "{} {} {}".format(m[0][0], m[0][1], m[1])
            ).saveAsTextFile(path)

            result = path
        else:
            result = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )

        return result
    else:
        raise Exception("Unsupported type for first matrix!")
