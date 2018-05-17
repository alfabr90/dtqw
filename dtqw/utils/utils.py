import os
import sys
import math
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
    def get_conf(sc, config_str, default=None):
        c = sc.getConf().get(config_str)

        if not c and (default is not None):
            return default

        return c

    @staticmethod
    def change_coordinate(rdd, old_coord, new_coord=CoordinateDefault):
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
    def filename(mesh_filename, steps, num_particles):
        return "{}_{}_{}".format(mesh_filename, steps, num_particles)

    @staticmethod
    def get_precendent_type(type1, type2):
        if type1 == complex or type2 == complex:
            return complex

        if type1 == float or type2 == float:
            return float

        return int

    @staticmethod
    def get_size_of_type(data_type):
        return sys.getsizeof(data_type())

    @staticmethod
    def get_num_partitions(spark_context, expected_size):
        safety_factor = 1.3
        num_partitions = None

        if Utils.get_conf(spark_context, 'dtqw.useSparkDefaultPartitions', default='False') == 'False':
            num_cores = Utils.get_conf(spark_context, 'dtqw.cluster.totalCores', default=None)

            if not num_cores:
                raise ValueError("Invalid number of total cores in the cluster: {}".format(num_cores))

            num_cores = int(num_cores)
            max_partition_size = int(Utils.get_conf(spark_context, 'dtqw.cluster.maxPartitionSize', default=48 * 10 ** 6))
            num_partitions = math.ceil(safety_factor * expected_size / max_partition_size / num_cores) * num_cores

        return num_partitions

    @staticmethod
    def create_dir(path):
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError('"{}" is an invalid path'.format(path))
        else:
            os.makedirs(path)

    @staticmethod
    def get_temp_path(d):
        tmp_file = tf.NamedTemporaryFile(dir=d)
        tmp_file.close()

        return tmp_file.name

    @staticmethod
    def remove_path(path):
        if os.path.exists(path):
            if os.path.isdir(path):
                if path != '/':
                    for i in os.listdir(path):
                        Utils.remove_path(path + "/" + i)
                    os.rmdir(path)
            else:
                os.remove(path)

    @staticmethod
    def clear_path(path):
        if os.path.exists(path):
            if os.path.isdir(path):
                if path != '/':
                    for i in os.listdir(path):
                        Utils.remove_path(path + "/" + i)
            else:
                raise ValueError('"{}" is an invalid path'.format(path))

    @staticmethod
    def get_size_of_path(path):
        if os.path.isdir(path):
            size = 0
            for i in os.listdir(path):
                size += Utils.get_size_of_path(path + "/" + i)
            return size
        else:
            return os.stat(path).st_size
