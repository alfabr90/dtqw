import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from dtqw.mesh.mesh import is_mesh
from dtqw.linalg.matrix import Matrix

__all__ = ['PDF', 'is_pdf']


class PDF(Matrix):
    """Class for probability density function of the system."""

    def __init__(self, spark_context, rdd, shape, mesh, num_particles):
        """
        Build a top level object for operators and system states.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
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

        super().__init__(spark_context, rdd, shape)

        self._mesh = mesh
        self._num_particles = num_particles

    @property
    def mesh(self):
        return self._mesh

    def dump(self):
        raise NotImplementedError

    def multiply(self, other):
        """
        Multiply this matrix with another one.

        Parameters
        ----------
        other :obj:Operator or :obj:State
            An operator if multiplying another operator, State otherwise.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def sum(self, ind, round_precision=10):
        """
        Sum the probabilities of this PDF.

        Parameters
        ----------
        ind : int
            The index to get the probability value. Depends on the dimension of the mesh and the number of particles.
        round_precision : int, optional
            The precision used to round the value. Default is 10 decimal digits.

        Returns
        -------

        """
        n = self.data.filter(
            lambda m: m[ind] != float()
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, round_precision)

    def plot(self, filename, title, labels=None, **kwargs):
        """
        Plot the probabilities over the mesh.

        Parameters
        ----------
        filename: str
            The filename to save the plot.
        title: str
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
                self._logger.warning('it is only possible to plot one and two dimensional meshes')
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

            axes.set_title(title)
            axes.view_init(elev=50)

            # figure.set_size_inches(12.8, 12.8)
        else:
            if self._logger:
                self._logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        plt.savefig(filename, kwargs=kwargs)
        plt.cla()
        plt.clf()

        if self._logger:
            self._logger.info("plot in {}s".format((datetime.now() - t1).total_seconds()))


def is_pdf(obj):
    return isinstance(obj, PDF)
