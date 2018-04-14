import random

__all__ = ['BrokenLinks']


class BrokenLinks():
    """Top-level class for broken links."""

    def __init__(self, spark_context):
        """
        Build a top-level broken links object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        """
        self._spark_context = spark_context

    @property
    def spark_context(self):
        return self._spark_context

    def generate(self, num_edges):
        """
        Yield broken links for the mesh.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError


def is_broken_links(obj):
    """
    Check whether argument is a BrokenLinks object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a BrokenLinks object, False otherwise.

    """
    return isinstance(obj, BrokenLinks)
