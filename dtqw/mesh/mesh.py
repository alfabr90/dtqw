from dtqw.utils.logger import Logger

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    def __init__(self, spark_context, size, log_filename='./log.txt'):
        self._spark_context = spark_context
        self._logger = Logger(__name__, log_filename)

        self._size = self._define_size(size)

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def logger(self):
        return self._logger

    @property
    def size(self):
        return self._size

    def __str__(self):
        return self.__class__.__name__

    def _validate(self, size):
        return False

    def _define_size(self, size):
        return None

    def to_string(self):
        return self.__str__()

    def title(self):
        return self.__str__()

    def filename(self):
        return self.__str__()

    def axis(self):
        return None

    def is_1d(self):
        return False

    def is_2d(self):
        return False

    def check_steps(self, steps):
        return True

    def create_operator(self, storage_level):
        return None


def is_mesh(obj):
    return isinstance(obj, Mesh)
