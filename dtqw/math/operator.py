import math

from dtqw.math.base import Base
from dtqw.math.state import State, is_state
from dtqw.utils.utils import CoordinateDefault, CoordinateMultiplier, CoordinateMultiplicand

__all__ = ['Operator', 'is_operator']


class Operator(Base):
    """Class for the operators of quantum walks."""

    def __init__(self, spark_context, rdd, shape, coord_format=CoordinateDefault):
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

        self._coordinate_format = coord_format

    @property
    def coordinate_format(self):
        return self._coordinate_format

    def dump(self, path):
        """
        Dump this object's RDD into disk. This method automatically converts the coordinate format to the default.

        Parameters
        ----------
        path : str
            The path where the dumped RDD will be located at

        Returns
        -------
        None

        """
        if self._coordinate_format == CoordinateMultiplier:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[1][0], m[0], m[1][1])
            )
        elif self._coordinate_format == CoordinateMultiplicand:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[0], m[1][0], m[1][1])
            )
        else:
            rdd = self.data.map(
                lambda m: " ".join([str(e) for e in m])
            )

        rdd.saveAsTextFile(path)

    def kron(self, other, coord_format=CoordinateDefault):
        """
        Perform a tensor (Kronecker) product with another operator.

        Parameters
        ----------
        other : :obj:Operator
            The other operator.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Operator.CoordinateDefault.

        Returns
        -------
        :obj:Operator
            The resulting operator.

        """
        if not is_operator(other):
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(other)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(other)))

        other_shape = other.shape
        new_shape = (self._shape[0] * other_shape[0], self._shape[1] * other_shape[1])

        num_partitions = max(self.data.getNumPartitions(), other.data.getNumPartitions())

        rdd = self.data.map(
            lambda m: (0, m)
        ).join(
            other.data.map(
                lambda m: (0, m)
            ),
            numPartitions=num_partitions
        ).map(
            lambda m: (m[1][0], m[1][1])
        )

        # rdd = self.data.cartesian(
        #     other.data
        # )

        if coord_format == CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1] * other_shape[1] + m[1][1], (m[0][0] * other_shape[0] + m[1][0], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], (m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Operator.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2])
            )

        return Operator(self._spark_context, rdd, new_shape, coord_format)

    def norm(self):
        """
        Calculate the norm of this operator.

        Returns
        -------
        float
            The norm of this operator.

        """
        n = self.data.filter(
            lambda m: m[2] != complex()
        ).map(
            lambda m: m[2].real ** 2 + m[2].imag ** 2
        ).reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

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

        if coord_format == CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1], (m[0][0], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0], (m[0][1], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Operator.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )

        return Operator(self._spark_context, rdd, shape, coord_format)

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

    def multiply(self, other, coord_format=CoordinateDefault):
        """
        Multiply this operator with another one or with a system state.

        Parameters
        ----------
        other :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Operator.CoordinateDefault. Not applicable when multiplying a State.

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
