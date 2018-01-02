import numpy as np

from dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Grover2D']


class Grover2D(Coin2D):
    """Class that represents the 2-dimensional Grover coin."""

    def __init__(self, spark_context):
        """
        Build a 2-dimensional Grover coin object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.

        """
        super().__init__(spark_context)

        self._data = np.array(
            [[-1, 1, 1, 1],
             [1, -1, 1, 1],
             [1, 1, -1, 1],
             [1, 1, 1, -1]], dtype=complex
        ) / 2.0
