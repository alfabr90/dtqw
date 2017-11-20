__all__ = ['Mesh', 'is_mesh']


class Mesh:
    def __init__(self, spark_context, size, bl_prob=None):
        self._spark_context = spark_context
        self._size = self._define_size(size)
        self._broken_links_probability = bl_prob

        self.logger = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def size(self):
        return self._size

    @property
    def broken_links_probability(self):
        return self._broken_links_probability

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def title(self):
        return self.__str__()

    def filename(self):
        return self.__str__()

    def _validate(self, size):
        return False

    def _define_size(self, size):
        return None

    def axis(self):
        return None

    def is_1d(self):
        return False

    def is_2d(self):
        return False

    def check_steps(self, steps):
        return True

    def create_operator(self):
        return None


def is_mesh(obj):
    return isinstance(obj, Mesh)
