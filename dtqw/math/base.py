import numpy as np
from pyspark import RDD, StorageLevel

from dtqw.utils.logger import is_logger
from dtqw.utils.profiling.profiler import is_profiler
from dtqw.utils.utils import Utils

__all__ = ['Base']


class Base:
    """Top-level class for some mathematical elements."""

    def __init__(self, rdd, shape, data_type=complex):
        """
        Build a top-level object for some mathematical elements.

        Parameters
        ----------
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default is complex.

        """
        if not isinstance(rdd, RDD):
            # self.logger.error("Invalid argument to instantiate an Operator object")
            raise TypeError("invalid argument to instantiate an Operator object")

        if shape is not None:
            if not Utils.is_shape(shape):
                # self.logger.error("Invalid shape")
                raise ValueError("invalid shape")

        self._spark_context = rdd.context
        self._shape = shape
        self._num_elements = self._shape[0] * self._shape[1]
        self._num_nonzero_elements = 0
        self._data_type = data_type

        self.data = rdd

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
    def data_type(self):
        return self._data_type

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

    def sparsity(self):
        """
        Calculate the sparsity of this object.

        Returns
        -------
        float
            The sparsity of this object.

        """
        return 1.0 - self.num_nonzero_elements / self._num_elements

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Persist this object's RDD considering the chosen storage level.

        Parameters
        ----------
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        :obj
            A reference to this object.

        """
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
        """
        Unpersist this object's RDD.

        Returns
        -------
        :obj
            A reference to this object.

        """
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
        """
        Alias of the method unpersist.

        Returns
        -------
        :obj
            A reference to this object.

        """
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Materialize this object's RDD considering the chosen storage level.

        This method calls persist and right after counts how many elements there are in the RDD to force its
        persistence.

        Parameters
        ----------
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        :obj
            A reference to this object.

        """
        self.persist(storage_level)
        self._num_nonzero_elements = self.data.count()

        if self._logger:
            self._logger.info("RDD {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        """
        Checkpoint this object's RDD.

        Returns
        -------
        :obj
            A reference to this object.

        """
        if self.data.isCheckpointed():
            if self._logger:
                self._logger.info("RDD already checkpointed")
            return self

        if not self.data.is_cached:
            if self._logger:
                self._logger.warning("it is recommended to cache the RDD before checkpointing it")

        self.data.checkpoint()

        if self._logger:
            self._logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

        return self

    def dump(self, path):
        """
        Dump this object's RDD into disk.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at

        Returns
        -------
        None

        """
        self.data.map(
            lambda m: " ".join([str(e) for e in m])
        ).saveAsTextFile(path)

    def numpy_array(self):
        """
        Create a numpy array containing this object's RDD data.

        Returns
        -------
        :obj:ndarray
            The numpy array

        """
        data = self.data.collect()
        ind = len(data[0]) - 1
        result = np.zeros(self._shape, dtype=self._data_type)

        for e in data:
            result[e[0:ind]] = e[ind]

        return result

    def is_unitary(self):
        """
        Check if this matrix is unitary by calculating its norm.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        round_precision = int(Utils.getConf(self._spark_context, 'dtqw.math.roundPrecision', default='10'))
        
        return round(self.norm(), round_precision) == 1.0
