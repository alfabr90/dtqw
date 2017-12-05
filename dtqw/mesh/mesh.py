from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.logger import is_logger
from dtqw.utils.profiler import is_profiler

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    def __init__(self, spark_context, size, bl_prob=None):
        self._spark_context = spark_context
        self._size = self._define_size(size)
        self._broken_links_probability = bl_prob

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def size(self):
        return self._size

    @property
    def broken_links_probability(self):
        return self._broken_links_probability

    @property
    def logger(self):
        return self._logger

    @property
    def profiler(self):
        return self._profiler

    @logger.setter
    def logger(self, logger):
        """
        Parameters
        ----------
        logger : Logger
            A Logger object or None to disable logging.

        Raises
        ------
        TypeError

        """
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError('logger instance expected, not "{}"'.format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        """
        Parameters
        ----------
        profiler : Profiler
            A Profiler object or None to disable profiling.

        Raises
        ------
        TypeError

        """
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError('profiler instance expected, not "{}"'.format(type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def _validate(self, size):
        raise NotImplementedError

    def _define_size(self, size):
        raise NotImplementedError

    def _profile(self, operator, initial_time):
        if self._profiler is None:
            return

        app_id = self._spark_context.applicationId
        rdd_id = operator.data.id()

        self._profiler.profile_times('shiftOperator', (datetime.now() - initial_time).total_seconds())
        self._profiler.profile_sparsity('shiftOperator', operator)
        self._profiler.profile_rdd('shiftOperator', app_id, rdd_id)
        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        if self._logger:
            self._logger.info(
                "shift operator was built in {}s".format(self._profiler.get_times(name='shiftOperator'))
            )
            self._logger.info(
                "shift operator is consuming {} bytes in memory and {} bytes in disk".format(
                    self._profiler.get_rdd(name='shiftOperator', key='memoryUsed'),
                    self._profiler.get_rdd(name='shiftOperator', key='diskUsed')
                )
            )
            self._logger.debug("shape of shift operator: {}".format(operator.shape))
            self._logger.debug(
                "number of elements of shift operator: {}, which {} are nonzero".format(
                    operator.num_elements, operator.num_nonzero_elements
                )
            )
            self._logger.debug("sparsity of shift operator: {}".format(operator.sparsity))

    def to_string(self):
        return self.__str__()

    def title(self):
        return self.__str__()

    def filename(self):
        return self.__str__()

    def axis(self):
        raise NotImplementedError

    def is_1d(self):
        raise NotImplementedError

    def is_2d(self):
        raise NotImplementedError

    def check_steps(self, steps):
        raise NotImplementedError

    def create_operator(self, num_partitions, mul_format=True, storage_level=StorageLevel.MEMORY_AND_DISK):
        raise NotImplementedError


def is_mesh(obj):
    return isinstance(obj, Mesh)
