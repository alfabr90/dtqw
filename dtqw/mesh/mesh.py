from datetime import datetime

from pyspark import StorageLevel

from dtqw.mesh.broken_links.broken_links import is_broken_links
from dtqw.utils.logger import is_logger
from dtqw.utils.profiling.profiler import is_profiler
from dtqw.utils.utils import Utils

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    """Top-level class for Meshes."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a top-level Mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int or tuple
            Size of the mesh.
        broken_links : BrokenLinks, optional
            A BrokenLinks object.
        """
        self._spark_context = spark_context
        self._size = self._define_size(size)
        self._num_edges = self._define_num_edges(size)

        if broken_links:
            if not is_broken_links(broken_links):
                # self.logger.error('broken links instance expected, not "{}"'.format(type(broken_links)))
                raise TypeError('broken links instance expected, not "{}"'.format(type(broken_links)))

        self._broken_links = broken_links

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def size(self):
        return self._size

    @property
    def num_edges(self):
        return self._num_edges

    @property
    def broken_links(self):
        return self._broken_links

    @property
    def logger(self):
        return self._logger

    @property
    def profiler(self):
        return self._profiler

    @logger.setter
    def logger(self, logger):
        """
        Parameters
        ----------
        logger : Logger
            A Logger object or None to disable logging.

        Raises
        ------
        TypeError

        """
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError('logger instance expected, not "{}"'.format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        """
        Parameters
        ----------
        profiler : Profiler
            A Profiler object or None to disable profiling.

        Raises
        ------
        TypeError

        """
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError('profiler instance expected, not "{}"'.format(type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def _validate(self, size):
        raise NotImplementedError

    def _define_size(self, size):
        raise NotImplementedError

    def _define_num_edges(self, size):
        raise NotImplementedError

    def _profile(self, operator, initial_time):
        if self._profiler is not None:
            app_id = self._spark_context.applicationId

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                'shiftOperator', operator, (datetime.now() - initial_time).total_seconds()
            )

            if self._logger:
                self._logger.info("shift operator was built in {}s".format(info['buildingTime']))
                self._logger.info(
                    "shift operator is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

    def to_string(self):
        return self.__str__()

    def title(self):
        return self.__str__()

    def filename(self):
        return self.__str__()

    def axis(self):
        raise NotImplementedError

    def is_1d(self):
        """
        Check if this is a 1-dimensional Mesh.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def is_2d(self):
        """
        Check if this is a 2-dimensional Mesh.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def check_steps(self, steps):
        """
        Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def create_operator(self, coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the mesh operator.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError


def is_mesh(obj):
    """
    Check whether argument is a Mesh object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a Mesh object, False otherwise.

    """
    return isinstance(obj, Mesh)
