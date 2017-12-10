import math
from pyspark import RDD, StorageLevel
from dtqw.utils.utils import is_shape
from dtqw.utils.logger import is_logger
from dtqw.utils.profiler import is_profiler

__all__ = ['Matrix']


class Matrix:
    """Top level class for operators and system states."""

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

    def __init__(self, spark_context, rdd, shape):
        """
        Build a top level object for operators and system states.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.

        """
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
        """
        Dump all this object's RDD to disk.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Persist this object's RDD considering the chosen storage level.

        Parameters
        ----------
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        Matrix
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
        Matrix
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
        Matrix
            A reference to this object.

        """
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Materialize this object's RDD considering the chosen storage level.

        This method calls persist and right after counts how many elements there are in the RDD to force its
        persistence and find the sparsity of the matrix.

        Parameters
        ----------
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        Matrix
            A reference to this object.

        """
        self.persist(storage_level)
        self._num_nonzero_elements = self.data.count()
        self._sparsity = 1.0 - self.num_nonzero_elements / self._num_elements

        if self._logger:
            self._logger.info("RDD {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        """
        Checkpoint this object's RDD.

        Returns
        -------
        Matrix
            A reference to this object.

        """
        if not self.data.is_cached:
            if self._logger:
                self._logger.warning("it is recommended to cache the RDD before checkpointing it")

        self.data.checkpoint()

        if self._logger:
            self._logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

        return self

    def norm(self):
        """
        Calculate the norm of this matrix.

        Returns
        -------
        float
            The norm of this matrix.

        """
        n = self.data.filter(
            lambda m: m[2] != complex()
        ).map(
            lambda m: m[2].real ** 2 + m[2].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self, round_precision=10):
        """
        Check if this matrix is unitary by calculating its norm.

        Parameters
        ----------
        round_precision : int, optional
            The desired precision when rounding the norm of this matrix. The default value is 10.

        Returns
        -------
        bool
            True if the norm of this matrix is 1.0, False otherwise.

        """
        return round(self.norm(), round_precision) == 1.0

    def multiply(self, other):
        """
        Multiply this matrix with another one.

        Parameters
        ----------
        other :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError
