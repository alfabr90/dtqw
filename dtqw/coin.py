import math

import numpy as np
import scipy.sparse as sp

from .math.operator import Operator
from .mesh import is_mesh
from dtqw.utils.logger import Logger

__all__ = ['Coin', 'is_coin',
           'HADAMARD_1D', 'HADAMARD_2D', 'GROVER_2D', 'FOURIER_2D']


HADAMARD_1D = 0
HADAMARD_2D = 1
GROVER_2D = 2
FOURIER_2D = 3


class Coin:
    def __init__(self, type, log_filename='log.txt'):
        self.__type = type
        self.__data = self.__define_data()
        self.__logger = Logger(__name__, log_filename)

    @property
    def type(self):
        return self.__type

    @property
    def data(self):
        return self.__data

    def __validate(self):
        return self.is_1d() or self.is_2d()

    def __define_data(self):
        if not self.__validate():
            self.__logger.error("Invalid coin")
            raise ValueError("invalid coin")

        if self.__type == HADAMARD_1D:
            return sp.coo_matrix(
                np.array(
                    [[1, 1],
                     [1, -1]], dtype=complex
                ) / math.sqrt(2)
            )
        elif self.__type == HADAMARD_2D:
            return sp.coo_matrix(
                np.array(
                    [[1, 1, 1, 1],
                     [1, -1, 1, -1],
                     [1, 1, -1, -1],
                     [1, -1, -1, 1]], dtype=complex
                ) / 2.0
            )
        elif self.__type == GROVER_2D:
            return sp.coo_matrix(
                np.array(
                    [[-1, 1, 1, 1],
                     [1, -1, 1, 1],
                     [1, 1, -1, 1],
                     [1, 1, 1, -1]], dtype=complex
                ) / 2.0
            )
        elif self.__type == FOURIER_2D:
            return sp.coo_matrix(
                np.array(
                    [[1, 1, 1, 1],
                     [1, 1.0j, -1, -1.0j],
                     [1, -1, 1, -1],
                     [1, -1.0j, -1, 1.0j]], dtype=complex
                ) / 2.0
            )

    def is_1d(self):
        return self.__type == HADAMARD_1D

    def is_2d(self):
        return self.__type == HADAMARD_2D or self.__type == GROVER_2D or self.__type == FOURIER_2D

    def create_operator(self, mesh, spark_context, log_filename='log.txt'):
        if not is_mesh(mesh):
            self.__logger.error('Expected mesh, not "{}"'.format(type(mesh)))
            raise TypeError('expected mesh, not "{}"'.format(type(mesh)))

        if mesh.is_1d():
            return Operator(sp.kron(self.__data, sp.identity(mesh.size)), spark_context, log_filename=log_filename)
        elif mesh.is_2d():
            return Operator(
                sp.kron(self.__data, sp.identity(mesh.size[0] * mesh.size[1])),
                spark_context,
                log_filename=log_filename
            )
        else:
            self.__logger.error("Mesh dimension not implemented")
            raise NotImplementedError


def is_coin(obj):
    return isinstance(obj, Coin)
