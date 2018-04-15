import math
import numpy as np

from dtqw.math.base import Base
from dtqw.math.state import State, is_state
from dtqw.utils.utils import Utils

__all__ = ['Operator', 'is_operator']


class Operator(Base):
    """Class for the operators of quantum walks."""

    def __init__(self, rdd, shape, data_type=complex, coord_format=Utils.CoordinateDefault):
        """
        Build an Operator object.

        Parameters
        ----------
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this operator object. Must be a 2-dimensional tuple.
        data_type : type, optional
            The Python type of all values in this object. Default is complex.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.
        """
        super().__init__(rdd, shape, data_type=data_type)

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
        if self._coordinate_format == Utils.CoordinateMultiplier:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[1][0], m[0], m[1][1])
            )
        elif self._coordinate_format == Utils.CoordinateMultiplicand:
            rdd = self.data.map(
                lambda m: "{}, {}, {}".format(m[0], m[1][0], m[1][1])
            )
        else:  # Utils.CoordinateDefault
            rdd = self.data.map(
                lambda m: " ".join([str(e) for e in m])
            )

        rdd.saveAsTextFile(path)

    def numpy_array(self):
        """
        Create a numpy array containing this object's RDD data.

        Returns
        -------
        :obj:ndarray
            The numpy array

        """
        data = self.data.collect()
        result = np.zeros(self._shape, dtype=self._data_type)

        if self._coordinate_format == Utils.CoordinateMultiplier:
            for e in data:
                result[e[1][0], e[0]] = e[1][1]
        elif self._coordinate_format == Utils.CoordinateMultiplicand:
            for e in data:
                result[e[0], e[1][0]] = e[1][1]
        else:  # Utils.CoordinateDefault
            for e in data:
                result[e[0], e[1]] = e[2]

        return result

    def kron(self, other, coord_format=Utils.CoordinateDefault):
        """
        Perform a tensor (Kronecker) product with another operator.

        Parameters
        ----------
        other : :obj:Operator
            The other operator.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.

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
        data_type = Utils.getPrecendentType(self._data_type, other.data_type)

        expected_elems = self._num_nonzero_elements * other.num_nonzero_elements
        expected_size = Utils.getSizeOfType(data_type) * expected_elems
        num_partitions = Utils.getNumPartitions(self.data.context, expected_size)

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

        if coord_format == Utils.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1] * other_shape[1] + m[1][1], (m[0][0] * other_shape[0] + m[1][0], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == Utils.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], (m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0] * other_shape[0] + m[1][0], m[0][1] * other_shape[1] + m[1][1], m[0][2] * m[1][2])
            )

        return Operator(rdd, new_shape, coord_format=coord_format)

    def norm(self):
        """
        Calculate the norm of this operator.

        Returns
        -------
        float
            The norm of this operator.

        """
        if self._coordinate_format == Utils.CoordinateMultiplier or self._coordinate_format == Utils.CoordinateMultiplicand:
            n = self.data.filter(
                lambda m: m[1][1] != complex()
            ).map(
                lambda m: m[1][1].real ** 2 + m[1][1].imag ** 2
            )
        else:  # Utils.CoordinateDefault
            n = self.data.filter(
                lambda m: m[2] != complex()
            ).map(
                lambda m: m[2].real ** 2 + m[2].imag ** 2
            )

        n = n.reduce(
            lambda a, b: a + b
        )

        return math.sqrt(n)

    def is_unitary(self):
        """
        Check if this operator is unitary by calculating its norm.

        Returns
        -------
        bool
            True if the norm of this operator is 1.0, False otherwise.

        """
        round_precision = int(Utils.getConf(self._spark_context, 'dtqw.math.roundPrecision', default='10'))

        return round(self.norm(), round_precision) == 1.0

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

        if coord_format == Utils.CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[0][1], (m[0][0], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        elif coord_format == Utils.CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0][0], (m[0][1], m[1]))
            ).partitionBy(
                numPartitions=num_partitions
            )
        else:  # Utils.CoordinateDefault
            rdd = rdd.map(
                lambda m: (m[0][0], m[0][1], m[1])
            )

        return Operator(rdd, shape, coord_format=coord_format)

    def _multiply_state(self, other):
        if self._shape[1] != other.shape[0]:
            if self._logger:
                self._logger.error("incompatible shapes {} and {}".format(self._shape, other.shape))
            raise ValueError("incompatible shapes {} and {}".format(self._shape, other.shape))

        shape = other.shape

        rdd = self.data.join(
            other.data, numPartitions=self.data.getNumPartitions()
        ).map(
            lambda m: (m[1][0][0], m[1][0][1] * m[1][1])
        ).reduceByKey(
            lambda a, b: a + b, numPartitions=self.data.getNumPartitions()
        )

        return State(rdd, shape, other.mesh, other.num_particles)

    def multiply(self, other, coord_format=Utils.CoordinateDefault):
        """
        Multiply this operator with another one or with a system state.

        Parameters
        ----------
        other :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault. Not applicable when multiplying a State.

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
