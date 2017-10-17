__all__ = ['Coin', 'is_coin']


class Coin:
    def __init__(self, spark_context):
        self._spark_context = spark_context
        self._size = None
        self._data = None

        self.logger = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def size(self):
        return self._size

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
