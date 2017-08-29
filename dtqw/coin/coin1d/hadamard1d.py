import math
import numpy as np
from dtqw.coin.coin1d.coin1d import Coin1D

__all__ = ['Hadamard1D']


class Hadamard1D(Coin1D):
    def __init__(self, spark_context, log_filename='./log.txt'):
        super().__init__(spark_context, log_filename)

        self._data = np.array(
            [[1, 1],
             [1, -1]], dtype=complex
        ) / math.sqrt(2)
