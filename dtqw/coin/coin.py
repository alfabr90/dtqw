from dtqw.utils.logger import Logger

__all__ = ['Coin', 'is_coin']


class Coin:
    def __init__(self, spark_context, log_filename='./log.txt'):
        self._spark_context = spark_context
        self._logger = Logger(__name__, log_filename)

        self._data = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def logger(self):
        return self._logger

    @property
    def data(self):
        return self._data

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def is_1d(self):
        return False

    def is_2d(self):
        return False

    def create_operator(self, mesh):
        return None


def is_coin(obj):
    return isinstance(obj, Coin)
