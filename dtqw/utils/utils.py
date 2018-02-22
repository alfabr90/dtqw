import os
import sys
import tempfile as tf

__all__ = ['Utils']


class Utils():
    CoordinateDefault = 0
    """
    CoordinateDefault : int
        Indicate that the Operator object must have its entries stored as (i,j,value) coordinates.
    """
    CoordinateMultiplier = 1
    """
    CoordinateMultiplier : int
        Indicate that the Operator object must have its entries stored as (j,(i,value)) coordinates. This is mandatory
        when the object is the multiplier operand.
    """
    CoordinateMultiplicand = 2
    """
    CoordinateMultiplicand : int
        Indicate that the Operator object must have its entries stored as (i,(j,value)) coordinates. This is mandatory
        when the object is the multiplicand operand.
    """

    def __init__(self):
        pass

    @staticmethod
    def is_shape(shape):
        return isinstance(shape, (list, tuple))

    @staticmethod
    def broadcast(sc, data):
        return sc.broadcast(data)

    @staticmethod
    def getConf(sc, config_str, default=None):
        c = sc.getConf().get(config_str)

        if not c and (default is not None):
            return default

        return c

    @staticmethod
    def changeCoordinate(rdd, old_coord, new_coord=CoordinateDefault):
        if old_coord == Utils.CoordinateMultiplier:
            if new_coord == Utils.CoordinateMultiplier:
                return rdd
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            else:  # Utils.CoordinateDefault
                return rdd.map(
                    lambda m: (m[1][0], m[0], m[1][1])
                )
        elif old_coord == Utils.CoordinateMultiplicand:
            if new_coord == Utils.CoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1][0], (m[0], m[1][1]))
                )
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd
            else:  # Utils.CoordinateDefault
                return rdd.map(
                    lambda m: (m[0], m[1][0], m[1][1])
                )
        else:  # Utils.CoordinateDefault
            if new_coord == Utils.CoordinateMultiplier:
                return rdd.map(
                    lambda m: (m[1], (m[0], m[2]))
                )
            elif new_coord == Utils.CoordinateMultiplicand:
                return rdd.map(
                    lambda m: (m[0], (m[1], m[2]))
                )
            else:  # Utils.CoordinateDefault
                return rdd

    @staticmethod
    def filename(mesh_filename, steps, num_particles, num_partitions):
        return "{}_{}_{}_{}".format(mesh_filename, steps, num_particles, num_partitions)

    @staticmethod
    def getPrecendentType(type1, type2):
        if type1 == complex or type2 == complex:
            return complex

        if type1 == float or type2 == float:
            return float

        return int

    @staticmethod
    def getSizeOfType(data_type):
        return sys.getsizeof(data_type())

    @staticmethod
    def createDir(path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError('"{}" is an invalid path'.format(path))
        else:
            os.makedirs(path)

    @staticmethod
    def getTempPath(d):
        tmp_file = tf.NamedTemporaryFile(dir=d)
        tmp_file.close()

        return tmp_file.name

    @staticmethod
    def removePath(path):
        if os.path.exists(path):
            if os.path.isdir(path):
                if path != '/':
                    for i in os.listdir(path):
                        Utils.removePath(path + "/" + i)
                    os.rmdir(path)
            else:
                os.remove(path)

    @staticmethod
    def clearPath(path):
        if os.path.exists(path):
            if os.path.isdir(path):
                if path != '/':
                    for i in os.listdir(path):
                        Utils.removePath(path + "/" + i)
            else:
                raise ValueError('"{}" is an invalid path'.format(path))

    @staticmethod
    def getSizeOfPath(path):
        if os.path.isdir(path):
            size = 0
            for i in os.listdir(path):
                size += Utils.getSizeOfPath(path + "/" + i)
            return size
        else:
            return os.stat(path).st_size
