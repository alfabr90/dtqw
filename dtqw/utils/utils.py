import io
import os
import tempfile as tf
import operator as op
import math
import csv
import numpy as np
import scipy.sparse as sp

from sys import getsizeof


STR_FORMAT_INT = 7
STR_FORMAT_FLOAT = 23

SAVE_MODE_MEMORY = 2 ** 0
SAVE_MODE_DISK = 2 ** 1
# SAVE_MODE_RDD = 2 ** 2

MUL_BROADCAST = 0
MUL_RDD = 1
MUL_BLOCK = 2

ROUND_PRECISION = 10

DEBUG_MODE = True


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


def coin_space(size, format='csc'):
    return sp.identity(size, format=format)


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


def is_shape(shape):
    return isinstance(shape, (list, tuple))


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
    if path is not None:
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


def export_times(execution_times, filename, extension='csv', path=None):
    if extension == 'csv':
        if path is None:
            path = './'
        else:
            create_dir(path)

        f = path + filename + "_TIMES." + extension

        fieldnames = execution_times.keys()

        with open(f, 'w') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            f.write(','.join(fieldnames) + "\n")
            w.writerow(execution_times)
    else:
        raise Exception("Unsupported file extension!")


def export_memory(memory_usage, filename, extension='csv', path=None):
    if extension == 'csv':
        if path is None:
            path = './'
        else:
            create_dir(path)

        f = path + filename + "_MEMORY." + extension

        fieldnames = memory_usage.keys()

        with open(f, 'w') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            f.write(','.join(fieldnames) + "\n")
            w.writerow(memory_usage)
    else:
        raise Exception("Unsupported file extension!")


def broadcast(sc, data):
    return sc.broadcast(data)


def convert_sparse(sparse, format):
    if format == 'csc':
        return sparse.tocsc()
    elif format == 'csr':
        return sparse.tocsr()
    elif format == 'lil':
        return sparse.tolil()
    elif format == 'dia':
        return sparse.todia()
    elif format == 'coo':
        return sparse.tocoo()
    elif format == 'bsr':
        return sparse.tobsr()
    elif format == 'dense':
        return sparse.todense()
    else:
        raise NotImplementedError


def build_initial_state(mesh, amplitudes, position,
                        entangled=False, coin_space_indices=None, operator=op.add, num_particles=1):
    if num_particles < 1:
        raise Exception("There must be at least one particle!")

    cs = coin_space(2)

    if mesh.is_1d():
        s = space(position, mesh.size)

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
    elif mesh.is_2d():
        sx, sy = space(position[0], mesh.size[0]), space(position[1], mesh.size[1])

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
