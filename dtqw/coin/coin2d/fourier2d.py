import numpy as np
from dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Fourier2D']


class Fourier2D(Coin2D):
    def __init__(self, spark_context):
        super().__init__(spark_context)

        self._data = np.array(
            [[1, 1, 1, 1],
             [1, 1.0j, -1, -1.0j],
             [1, -1, 1, -1],
             [1, -1.0j, -1, 1.0j]], dtype=complex
        ) / 2.0
