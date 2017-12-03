import math
from pyspark import RDD, StorageLevel
from dtqw.utils.utils import is_shape
from dtqw.utils.logger import is_logger
from dtqw.utils.profiler import is_profiler

__all__ = ['Matrix']


class Matrix:
    def __init__(self, spark_context, rdd, shape):
        self._spark_context = spark_context

        if not isinstance(rdd, RDD):
            # self.logger.error("Invalid argument to instantiate an Operator object")
            raise TypeError("invalid argument to instantiate an Operator object")

        self.data = rdd

        if shape is not None:
            if not is_shape(shape):
                # self.logger.error("Invalid shape")
                raise ValueError("invalid shape")

        self._shape = shape
        self._num_elements = self._shape[0] * self._shape[1]
        self._num_nonzero_elements = 0
        self._sparsity = 1

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def shape(self):
        return self._shape

    @property
    def num_elements(self):
        return self._num_elements

    @property
    def num_nonzero_elements(self):
        return self._num_nonzero_elements

    @property
    def sparsity(self):
        return self._sparsity

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

        Returns
        -------
        None

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

        Returns
        -------
        None

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

    def to_string(self):
        return self.__str__()

    def dump(self):
        raise NotImplementedError

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.data is not None:
            if not self.data.is_cached:
                self.data.persist(storage_level)
                if self._logger:
                    self._logger.info("RDD {} was persisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("RDD {} has already been persisted".format(self.data.id()))
        else:
            if self._logger:
                self._logger.warning("there is no data to be persisted")

        return self

    def unpersist(self):
        if self.data is not None:
            if self.data.is_cached:
                self.data.unpersist()
                if self._logger:
                    self._logger.info("RDD {} was unpersisted".format(self.data.id()))
            else:
                if self._logger:
                    self._logger.info("RDD has already been unpersisted".format(self.data.id()))
        else:
            if self._logger:
                self._logger.warning("there is no data to be unpersisted")

        return self

    def destroy(self):
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self.persist(storage_level)
        self._num_nonzero_elements = self.data.count()
        self._sparsity = 1.0 - self.num_nonzero_elements / self._num_elements

        if self._logger:
            self._logger.info("RDD {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        if not self.data.is_cached:
            if self._logger:
                self._logger.warning("it is recommended to cache the RDD before checkpointing it")

        self.data.checkpoint()

        if self._logger:
            self._logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

        return self

    def is_unitary(self, round_precision=10):
        n = self.data.filter(
            lambda m: m[2] != complex()
        ).map(
            lambda m: m[2].real ** 2 + m[2].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return round(math.sqrt(n), round_precision) == 1.0

    def multiply(self, other):
        raise NotImplementedError
