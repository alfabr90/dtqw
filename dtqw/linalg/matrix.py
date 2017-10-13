import math
from pyspark import RDD, StorageLevel
from dtqw.utils.utils import is_shape

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

        self.logger = None
        self.profiler = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def shape(self):
        return self._shape

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK_SER):
        if self.data is not None:
            if not self.data.is_cached:
                self.data.persist(storage_level)
                if self.logger:
                    self.logger.info("RDD {} was persisted".format(self.data.id()))
            else:
                if self.logger:
                    self.logger.info("RDD {} has already been persisted".format(self.data.id()))
        else:
            if self.logger:
                self.logger.warning("there is no data to be persisted")

        return self

    def unpersist(self):
        if self.data is not None:
            if self.data.is_cached:
                self.data.unpersist()
                if self.logger:
                    self.logger.info("RDD {} was unpersisted".format(self.data.id()))
            else:
                if self.logger:
                    self.logger.info("RDD has already been unpersisted".format(self.data.id()))
        else:
            if self.logger:
                self.logger.warning("there is no data to be unpersisted")

        return self

    def destroy(self):
        return self.unpersist()

    def materialize(self, storage_level=StorageLevel.MEMORY_AND_DISK_SER):
        self.persist(storage_level)
        self.data.count()

        if self.logger:
            self.logger.info("RDD {} was materialized".format(self.data.id()))

        return self

    def checkpoint(self):
        if not self.data.is_cached:
            if self.logger:
                self.logger.warning("it is recommended to cache the RDD before checkpointing it")

        self.data.checkpoint()

        if self.logger:
            self.logger.info("RDD {} was checkpointed in {}".format(self.data.id(), self.data.getCheckpointFile()))

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
        return None
