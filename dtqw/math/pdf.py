import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from dtqw.mesh.mesh import is_mesh
from dtqw.math.matrix import Matrix

__all__ = ['PDF', 'is_pdf']


class PDF(Matrix):
    def __init__(self, spark_context, rdd, shape, mesh, num_particles, log_filename='./log.txt'):
        super().__init__(spark_context, rdd, shape, log_filename)

        if not is_mesh(mesh):
            self._logger.error('Mesh instance expected, not "{}"'.format(type(mesh)))
            raise TypeError('Mesh instance expected, not "{}"'.format(type(mesh)))

        self._mesh = mesh

        self._num_particles = num_particles

    @property
    def mesh(self):
        return self._mesh

    def sum(self, ind, round_precision=10):
        n = self.data.filter(
            lambda m: m[ind] != 0.0
        ).map(
            lambda m: m[ind]
        ).reduce(
            lambda a, b: a + b
        )

        return round(n, round_precision)

    def plot(self, title, labels, filename, path=None, **kwargs):
        self._logger.info("Start ploting probabilities...")

        if len(self._shape) > 2:
            self._logger.warning('It is only possible to plot one and two dimensional meshes')
            return None

        t1 = datetime.now()

        plt.cla()
        plt.clf()

        axis = self._mesh.axis()

        if self._mesh.is_1d():
            # self._build_onedim_plot(pdf.data, axis, labels, title)

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
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            plt.title(title)
        elif self._mesh.is_2d():
            # self._build_twodim_plot(pdf.data, axis, labels, title)

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

            axes.set_xlabel(labels[0])
            axes.set_ylabel(labels[1])
            axes.set_zlabel(labels[2])
            axes.set_title(title)
            axes.view_init(elev=50)

            # figure.set_size_inches(12.8, 12.8)
        else:
            self._logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        if path is None:
            plt.show()
        else:
            plt.savefig(path + filename, kwargs=kwargs)

        self._logger.info("Plots in {}s".format((datetime.now() - t1).total_seconds()))


def is_pdf(obj):
    return isinstance(obj, PDF)
