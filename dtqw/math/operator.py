from dtqw.utils.utils import get_tmp_path
from dtqw.math.state import State, is_state
from dtqw.math.matrix import Matrix

__all__ = ['Operator', 'is_operator']


class Operator(Matrix):
    def __init__(self, spark_context, rdd, shape):
        super().__init__(spark_context, rdd, shape)

    def dump(self):
        path = get_tmp_path()

        self.data.map(
            lambda m: "{} {} {}".format(m[0], m[1], m[2])
        ).saveAsTextFile(path)

        if self.logger:
            self.logger.info("RDD {} was dumped to disk in {}".format(self.data.id(), path))

        self.data.unpersist()

        def __map(m):
            m = m.split()
            return int(m[0]), int(m[1]), complex(m[2])

        self.data = self._spark_context.textFile(
            path
        ).map(
            __map
        )

        return self

    def _multiply_operator(self, other):
        if self._shape[1] != other.shape[0]:
            if self.logger:
                self.logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError('incompatible shapes {} and {}'.format(self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.join(
            other.data, numPartitions=num_partitions
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        ).filter(
            lambda m: m[1] != complex
        ).map(
            lambda m: (m[0][1], m[0][0], m[1])
        )

        return Operator(self._spark_context, rdd, shape)

    def _multiply_state(self, other):
        if self._shape[1] != other.shape[0]:
            if self.logger:
                self.logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = other.shape

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.join(
            other.data, numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0][0], m[1][0][1] * m[1][1])
        ).filter(
            lambda m: m[1] != complex
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        )

        return State(self._spark_context, rdd, shape, other.mesh, other.num_particles)

    def multiply(self, other):
        if is_operator(other):
            return self._multiply_operator(other)
        elif is_state(other):
            return self._multiply_state(other)
        else:
            if self.logger:
                self.logger.error('State or Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('State or Operator instance expected, not "{}"'.format(type(other)))

    def kron(self, other_broadcast, other_shape):
        shape = (self._shape[0] * other_shape[0], self._shape[1] * other_shape[1])

        def __map(m):
            for i in other_broadcast.value:
                yield (m[0] * other_shape[0] + i[0], m[1] * other_shape[1] + i[1], m[2] * i[2])

        rdd = self.data.flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape)


def is_operator(obj):
    return isinstance(obj, Operator)
