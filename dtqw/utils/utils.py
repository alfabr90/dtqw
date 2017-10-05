import os
import tempfile as tf

__all__ = ['ROUND_PRECISION', 'is_shape', 'broadcast',
           'filename', 'create_dir', 'get_tmp_path', 'remove_tmp_path', 'size_of_tmp_path']


ROUND_PRECISION = 10


def is_shape(shape):
    return isinstance(shape, (list, tuple))


def broadcast(sc, data):
    return sc.broadcast(data)


def filename(mesh_filename, steps, num_particles, num_partitions):
    return "{}_{}_{}_{}".format(mesh_filename, steps, num_particles, num_partitions)


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
