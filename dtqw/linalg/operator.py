from dtqw.utils.utils import get_tmp_path
from dtqw.linalg.state import State, is_state
from dtqw.linalg.matrix import Matrix

__all__ = ['Operator', 'is_operator']


class Operator(Matrix):
    """Class for the operators of quantum walks."""

    def __init__(self, spark_context, rdd, shape):
        """
        Build an Operator object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this operator object. Must be a 2-dimensional tuple.
        """
        super().__init__(spark_context, rdd, shape)

    def dump(self):
        """
        Dump all this object's RDD to disk.

        Returns
        ------
        :obj:Operator
            A reference to this object.

        """
        path = get_tmp_path()

        self.data.map(
            lambda m: "{} {} {}".format(m[0], m[1], m[2])
        ).saveAsTextFile(path)

        if self._logger:
            self._logger.info("RDD {} was dumped to disk in {}".format(self.data.id(), path))

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

    def _multiply_operator(self, other, coord_format):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError('incompatible shapes {} and {}'.format(self._shape, other.shape))

        shape = (self._shape[0], other.shape[1])

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.join(
            other.data, numPartitions=num_partitions
        ).map(
            lambda m: ((m[1][0][0], m[1][1][0]), m[1][0][1] * m[1][1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        )

        if coord_format == Matrix.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1], (m[0][0], m[1]))
            )
        elif coord_format == Matrix.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0], (m[0][1], m[1]))
            )
        else:  # Matrix.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )

        return Operator(self._spark_context, rdd, shape)

    def _multiply_state(self, other):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = other.shape

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.join(
            other.data, numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0][0], m[1][0][1] * m[1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=num_partitions
        )

        return State(self._spark_context, rdd, shape, other.mesh, other.num_particles)

    def multiply(self, other, coord_format=Matrix.CoordinateDefault):
        """
        Multiply this operator with another one or a system state.

        Parameters
        ----------
        other :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Matrix.CoordinateDefault. Not applicable when multiplying a State.

        Returns
        -------
        :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.

        Raises
        ------
        TypeError
            If other is neither an operator nor a state.

        """
        if is_operator(other):
            return self._multiply_operator(other, coord_format)
        elif is_state(other):
            return self._multiply_state(other)
        else:
            if self._logger:
                self._logger.error('State or Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('State or Operator instance expected, not "{}"'.format(type(other)))

    def kron(self, other_broadcast, other_shape):
        """
        Perform the tensor product between this object and another operator.

        Parameters
        ----------
        other_broadcast : Broadcast
            A Spark's broadcast variable containing a collection of the other
            operator's entries in (i,j,value) coordinate.
        other_shape : tuple
            The shape of the other operator. Must be a 2-dimensional tuple.

        Returns
        -------
        :obj:Operator
            The resulting operator.

        """
        shape = (self._shape[0] * other_shape[0], self._shape[1] * other_shape[1])

        def __map(m):
            for i in other_broadcast.value:
                yield (m[0] * other_shape[0] + i[0], m[1] * other_shape[1] + i[1], m[2] * i[2])

        rdd = self.data.flatMap(
            __map
        )

        return Operator(self._spark_context, rdd, shape)


def is_operator(obj):
    """
    Check whether argument is an Operator object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is an Operator object, False otherwise.

    """
    return isinstance(obj, Operator)
