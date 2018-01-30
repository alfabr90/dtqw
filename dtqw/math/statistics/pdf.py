import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from dtqw.math.base import Base
from dtqw.mesh.mesh import is_mesh

__all__ = ['PDF', 'is_pdf']


class PDF(Base):
    """Top-level class for probability distribution functions."""

    def __init__(self, rdd, shape, mesh, num_particles):
        """
        Build a top-level object for probability distribution functions.

        Parameters
        ----------
        rdd : RDD
            The base RDD of this object.
        shape : tuple
            The shape of this matrix object. Must be a 2-dimensional tuple.
        mesh : Mesh
            The mesh where the particles will walk on.
        num_particles : int
            The number of particles present in the walk.

        """
        if not is_mesh(mesh):
            # self.logger.error('Mesh instance expected, not "{}"'.format(type(mesh)))
            raise TypeError('mesh instance expected, not "{}"'.format(type(mesh)))

        super().__init__(rdd, shape, data_type=float)

        self._mesh = mesh
        self._num_particles = num_particles

    @property
    def mesh(self):
        return self._mesh

    def sum(self):
        """
        Sum the values of this PDF.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def norm(self):
        """
        Calculate the norm of this operator.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def expected_value(self):
        """
        Calculate the expected value of this PDF.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def variance(self, mean=None):
        """
        Calculate the variance of this PDF.

        Parameters
        ----------
        mean : float, optional
            The mean of this PDF. When None is passed as argument, the mean is calculated.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def plot(self, filename, title=None, labels=None, **kwargs):
        """
        Plot the probabilities over the mesh.

        Parameters
        ----------
        filename: str
            The filename to save the plot.
        title: str, optional
            The title of the plot.
        labels: tuple or list, optional
            The labels of each axis.
        kwargs
            Keyword arguments being passed to matplotlib.

        Returns
        -------
        None

        """
        if self._logger:
            self._logger.info("starting plot of probabilities...")

        if len(self._shape) > 2:
            if self._logger:
                self._logger.warning('it is only possible to plot one and two-dimensional meshes')
            return None

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._mesh.axis()

        if self._mesh.is_1d():
            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0]] = i[1]

            plt.plot(
                axis,
                pdf,
                color='b',
                linestyle='-',
                linewidth=1.0
            )

            if labels:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            else:
                plt.xlabel('Position')
                plt.ylabel('Probability')

            if title:
                plt.title(title)
        elif self._mesh.is_2d():
            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0], i[1]] = i[2]

            figure = plt.figure()
            axes = figure.add_subplot(111, projection='3d')

            axes.plot_surface(
                axis[0],
                axis[1],
                pdf,
                rstride=1,
                cstride=1,
                cmap=plt.cm.YlGnBu_r,
                linewidth=0.1,
                antialiased=True
            )

            if labels:
                axes.set_xlabel(labels[0])
                axes.set_ylabel(labels[1])
                axes.set_zlabel(labels[2])
            else:
                axes.set_xlabel('Position x')
                axes.set_ylabel('Position y')
                axes.set_zlabel('Probability')

            if title:
                axes.set_title(title)
            axes.view_init(elev=50)

            # figure.set_size_inches(12.8, 12.8)
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        plt.savefig(filename, **kwargs)
        plt.cla()
        plt.clf()

        if self._logger:
            self._logger.info("plot in {}s".format((datetime.now() - t1).total_seconds()))

    def plot_contour(self, filename=None, title=None, labels=None, **kwargs):
        """
        Plot the contour function of the probabilities over the mesh.

        Parameters
        ----------
        filename: str
            The filename to save the plot.
        title: str, optional
            The title of the plot.
        labels: tuple or list, optional
            The labels of each axis.
        kwargs
            Keyword arguments being passed to matplotlib.

        Returns
        -------
        None

        """
        if self._logger:
            self._logger.info("starting contour plot of probabilities...")

        if len(self._shape) > 2:
            if self._logger:
                self._logger.warning('it is only possible to plot the contour function of two-dimensional meshes')
            return None

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._mesh.axis()

        if self._mesh.is_1d():
            if self._logger:
                self._logger.error("mesh dimension not implemented to contour plots")
            raise NotImplementedError("mesh dimension not implemented to contour plots")
        elif self._mesh.is_2d():
            pdf = np.zeros(self._shape, dtype=float)

            for i in self.data.collect():
                pdf[i[0], i[1]] = i[2]

            plt.contourf(axis[0], axis[1], pdf)
            plt.colorbar()

            if labels:
                plt.xlabel(labels[0])
                plt.ylabel(labels[1])
            else:
                plt.xlabel('Position x')
                plt.ylabel('Position y')

            if title:
                plt.title(title)

            # figure.set_size_inches(12.8, 12.8)
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        plt.savefig(filename, **kwargs)
        plt.cla()
        plt.clf()

        if self._logger:
            self._logger.info("contour plot in {}s".format((datetime.now() - t1).total_seconds()))


def is_pdf(obj):
    """
    Check whether argument is a PDF object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a PDF object, False otherwise.

    """
    return isinstance(obj, PDF)
