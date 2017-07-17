import cmath
from datetime import datetime
import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse as sp
from scipy.sparse.linalg import norm
from pyspark import RDD, StorageLevel

from .mesh import *
from .state import *
from .operator import *
from .metrics import *

from .logger import *
from .utils import SAVE_MODE_MEMORY, SAVE_MODE_DISK, get_tmp_path, remove_tmp_path, create_dir


class DiscreteTimeQuantumWalk:
    def __init__(self, spark_context, coin, mesh,
                 num_particles=1, min_partitions=8, save_mode=SAVE_MODE_MEMORY,
                 storage_level=StorageLevel.MEMORY_ONLY, log_filename='log.txt'):
        if num_particles < 1:
            raise Exception("There must be at least one particle!")

        self.__min_partitions = min_partitions
        self.__save_mode = save_mode
        self.__storage_level = storage_level

        self.__spark_context = spark_context
        self.__coin = coin
        self.__mesh = mesh
        self.__num_particles = num_particles

        self.__coin_operator = None
        self.__shift_operator = None
        self.__unitary_operator = None
        self.__interaction_operator = None
        self.__multiparticles_unitary_operator = None
        self.__walk_operator = None

        self.__steps = 0

        self.__logger = Logger(__name__, log_filename)
        self.__metrics = Metrics(log_filename=log_filename)

        self.__execution_times = {
            'coin_operator': 0.0,
            'shift_operator': 0.0,
            'unitary_operator': 0.0,
            'interaction_operator': 0.0,
            'multiparticles_unitary_operator': 0.0,
            'walk_operator': 0.0,
            'walk': 0.0,
            'export_plot': 0.0
        }

        self.__memory_usage = {
            'coin_operator': 0,
            'shift_operator': 0,
            'unitary_operator': 0,
            'interaction_operator': 0,
            'multiparticles_unitary_operator': 0,
            'walk_operator': 0,
            'state': 0
        }

    @property
    def spark_context(self):
        return self.__spark_context

    @property
    def coin(self):
        return self.__coin

    @property
    def mesh(self):
        return self.__mesh

    @property
    def memory_usage(self):
        return self.__memory_usage

    @property
    def execution_times(self):
        return self.__execution_times

    @property
    def coin_operator(self):
        return self.__coin_operator

    @property
    def shift_operator(self):
        return self.__shift_operator

    @property
    def unitary_operator(self):
        return self.__unitary_operator

    @property
    def interaction_operator(self):
        return self.__interaction_operator

    @property
    def multiparticles_unitary_operator(self):
        return self.__multiparticles_unitary_operator

    @property
    def walk_operator(self):
        return self.__walk_operator

    @property
    def memory_usage(self):
        return self.__memory_usage
    
    @property
    def execution_times(self):
        return self.__execution_times

    @coin_operator.setter
    def coin_operator(self, co):
        if is_operator(co):
            self.__coin_operator = co
        else:
            raise TypeError("Operator instance expected")

    @shift_operator.setter
    def shift_operator(self, so):
        if is_operator(so):
            self.__shift_operator = so
        else:
            raise TypeError("Operator instance expected")

    @unitary_operator.setter
    def unitary_operator(self, uo):
        if is_operator(uo):
            self.__unitary_operator = uo
        else:
            raise TypeError("Operator instance expected")

    @interaction_operator.setter
    def interaction_operator(self, io):
        if is_operator(io):
            self.__interaction_operator = io
        else:
            raise TypeError("Operator instance expected")

    @multiparticles_unitary_operator.setter
    def multiparticles_unitary_operator(self, mu):
        if is_operator(mu):
            self.__multiparticles_unitary_operator = mu
        else:
            raise TypeError("Operator instance expected")

    @walk_operator.setter
    def walk_operator(self, wo):
        if is_operator(wo):
            self.__walk_operator = wo
        else:
            raise TypeError("Operator instance expected")

    def create_coin_operator(self):
        self.__logger.info("Building coin operator...")
        t1 = datetime.now()

        result = self.__coin.create_operator(self.__mesh, self.__spark_context, log_filename=self.__logger.filename)

        self.__execution_times['coin_operator'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['coin_operator'] = result.memory_usage

        self.__logger.info("Coin operator was built in {}s".format(self.__execution_times['coin_operator']))
        self.__logger.info("Coin operator is consuming {} bytes".format(self.__memory_usage['coin_operator']))
        self.__logger.debug("Coin operator format: {}".format(result.format))
        if result.is_path():
            self.__logger.debug("Coin operator path: {}".format(result.data))
        self.__logger.debug("Shape of coin operator: {}".format(result.shape))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return result

    def create_shift_operator(self):
        self.__logger.info("Building shift operator...")
        t1 = datetime.now()

        result = self.__mesh.create_operator(
            self.__spark_context, self.__min_partitions, log_filename=self.__logger.filename
        )

        self.__execution_times['shift_operator'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['shift_operator'] = result.memory_usage

        self.__logger.info("Shift operator was built in {}s".format(self.__execution_times['shift_operator']))
        self.__logger.info("Shift operator is consuming {} bytes".format(self.__memory_usage['shift_operator']))
        self.__logger.debug("Shift operator format: {}".format(result.format))
        if result.is_path():
            self.__logger.debug("Shift operator path: {}".format(result))
        self.__logger.debug("Shape of shift operator: {}".format(result.shape))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return result

    def create_unitary_operator(self):
        self.__logger.info("Building unitary operator...")

        if self.__coin_operator is None:
            self.__logger.info("No coin operator has been set. A new one will be built")
            self.__coin_operator = self.create_coin_operator()

        if self.__shift_operator is None:
            self.__logger.info("No shift operator has been set. A new one will be built")
            self.__shift_operator = self.create_shift_operator()

        t1 = datetime.now()

        result = self.__shift_operator.multiply(self.__coin_operator, self.__min_partitions)
        result.to_path(self.__min_partitions)
        result.to_rdd(self.__min_partitions)
        result.materialize()

        self.__coin_operator.destroy()
        self.__shift_operator.destroy()

        self.__execution_times['unitary_operator'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['unitary_operator'] = result.memory_usage

        self.__logger.info("Unitary operator was built in {}s".format(self.__execution_times['unitary_operator']))
        self.__logger.info("Unitary operator is consuming {} bytes".format(self.__memory_usage['unitary_operator']))
        self.__logger.debug("Unitary operator format: {}".format(result.format))
        if result.is_path():
            self.__logger.info("Unitary operator path: {}".format(result))
        self.__logger.debug("Shape of unitary operator: {}".format(result.shape))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return result

    def create_interaction_operator(self, phase):
        self.__logger.info("Building interaction operator...")
        t1 = datetime.now()

        num_p = self.__num_particles
        cp = cmath.exp(phase * (0.0+1.0j))
        cs = 2

        if self.__mesh.is_1d():
            ndim = 1
            size = self.__mesh.size
            cs_size = cs * self.__mesh.size
            shape = (cs_size ** num_p, cs_size ** num_p)

            def __map(m):
                a = []
                for p in range(num_p):
                    a.append(int(m / (cs_size ** (num_p - 1 - p))) % size)
                for i in range(num_p):
                    if a[0] != a[i]:
                        return m, m, 1
                return m, m, cp
        elif self.__mesh.is_2d():
            ndim = 2
            ind = ndim * num_p
            size_x = self.__mesh.size[0]
            size_y = self.__mesh.size[1]
            cs_size_x = cs * self.__mesh.size[0]
            cs_size_y = cs * self.__mesh.size[1]
            shape = ((cs_size_x * cs_size_y) ** num_p, (cs_size_x * cs_size_y) ** num_p)

            def __map(m):
                a = []
                for p in range(num_p):
                    a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                    a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                for i in range(0, ind, ndim):
                    if a[0] != a[i] or a[1] != a[i + 1]:
                        return m, m, 1
                return m, m, cp

        rdd = self.__spark_context.range(
            shape[0], numSlices=self.__min_partitions
        ).map(
            __map
        )

        result = Operator(rdd, self.__spark_context, shape)
        result.clear_rdd_path()

        self.__execution_times['interaction_operator'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['interaction_operator'] = result.memory_usage

        self.__logger.info(
            "Interaction operator was built in {}s".format(self.__execution_times['interaction_operator'])
        )
        self.__logger.info(
            "Interaction operator is consuming {} bytes".format(self.__memory_usage['interaction_operator'])
        )
        self.__logger.debug("Interaction operator format: {}".format(result.format))
        if result.is_path():
            self.__logger.info("Interaction operator path: {}".format(result.data))
        self.__logger.debug("Shape of interaction operator: {}".format(result.shape))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return result

    def create_multiparticles_unitary_operator(self):
        self.__logger.info("Building multiparticles unitary operator...")

        if self.__unitary_operator is None:
            self.__logger.info("No unitary operator has been set. A new one will be built")
            self.__unitary_operator = self.create_unitary_operator()

        t1 = datetime.now()

        other = self.__unitary_operator.to_rdd(self.__min_partitions, True)
        other.clear_rdd_path()

        result = self.__unitary_operator  # .to_rdd(self.__min_partitions, True)

        for p in range(self.__num_particles - 1):
            result_tmp = result.kron(other, self.__min_partitions)
            result_tmp.to_path(self.__min_partitions)
            result_tmp.to_rdd(self.__min_partitions)
            result_tmp.materialize()
            result.destroy()
            result = result_tmp

        other.destroy()

        self.__execution_times['multiparticles_unitary_operator'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['multiparticles_unitary_operator'] = result.memory_usage

        self.__logger.info(
            "Multiparticles unitary operator was built in {}s".format(
                self.__execution_times['multiparticles_unitary_operator']
            )
        )
        self.__logger.info(
            "Multiparticles unitary operator is consuming {} bytes".format(
                self.__memory_usage['multiparticles_unitary_operator']
            )
        )
        self.__logger.debug("Multiparticles unitary operator format: {}".format(result.format))
        if result.is_path():
            self.__logger.info("Multiparticles unitary operator path: {}".format(result.data))
        self.__logger.debug("Shape of multiparticles unitary operator: {}".format(result.shape))

        app_id = self.__spark_context.applicationId
        self.__metrics.log_rdds(app_id=app_id)

        return result

    def create_walk_operator(self, collision_phase=None):
        if self.__num_particles == 1:
            self.__logger.info("With just one particle, the walk operator is the unitary operator")

            if self.__unitary_operator is None:
                self.__logger.info("No unitary operator has been set. A new one will be built")
                self.__unitary_operator = self.create_unitary_operator()

            result = self.__unitary_operator
        else:
            if collision_phase is None:
                self.__logger.info("No collision phase has been defined. "
                                   "The walk operator will be the multiparticles unitary operator")

                if self.__multiparticles_unitary_operator is None:
                    self.__logger.info("No multiparticles unitary operator has been set. A new one will be built")
                    self.__multiparticles_unitary_operator = self.create_multiparticles_unitary_operator()

                result = self.__multiparticles_unitary_operator
            else:
                self.__logger.info("Building walk operator...")

                if self.__multiparticles_unitary_operator is None:
                    self.__logger.info("No multiparticles unitary operator has been set. A new one will be built")
                    self.__multiparticles_unitary_operator = self.create_multiparticles_unitary_operator()

                if self.__interaction_operator is None:
                    self.__logger.info("No interaction operator has been set. A new one will be built")
                    self.__interaction_operator = self.create_interaction_operator(collision_phase)

                t1 = datetime.now()

                result = self.__multiparticles_unitary_operator.multiply(
                    self.__interaction_operator,
                    self.__min_partitions
                )
                result.to_path(self.__min_partitions)
                result.to_rdd(self.__min_partitions)
                result.materialize()

                self.__multiparticles_unitary_operator.destroy()
                self.__interaction_operator.destroy()

                self.__execution_times['walk_operator'] = (datetime.now() - t1).total_seconds()
                self.__memory_usage['walk_operator'] = result.memory_usage

                self.__logger.info(
                    "Walk operator was built in {}s".format(self.__execution_times['walk_operator'])
                )
                self.__logger.info(
                    "Walk operator is consuming {} bytes".format(self.__memory_usage['walk_operator'])
                )
                self.__logger.debug("Walk operator format: {}".format(result.format))
                if result.is_path():
                    self.__logger.info("Walk operator path: {}".format(result.data))
                self.__logger.debug("Shape of walk operator: {}".format(result.shape))

                app_id = self.__spark_context.applicationId
                self.__metrics.log_rdds(app_id=app_id)

        return result

    def plot_title(self):
        if self.__mesh.is_1d():
            if self.__mesh.type == MESH_1D_LINE:
                mesh = "Line"
            elif self.__mesh.type == MESH_1D_SEGMENT:
                mesh = "Segment"
            elif self.__mesh.type == MESH_1D_CYCLE:
                mesh = "Cycle"

            if self.__num_particles == 1:
                particles = str(self.__num_particles) + " Particle on a "
            else:
                particles = str(self.__num_particles) + " Particles on a "

            return "Quantum Walk with " + particles + mesh
        elif self.__mesh.is_2d():
            if self.__mesh.type == MESH_2D_LATTICE_DIAGONAL:
                mesh = "Diagonal Lattice"
            elif self.__mesh.type == MESH_2D_LATTICE_NATURAL:
                mesh = "Natural Lattice"
            elif self.__mesh.type == MESH_2D_BOX_DIAGONAL:
                mesh = "Diagonal Box"
            elif self.__mesh.type == MESH_2D_BOX_NATURAL:
                mesh = "Natural Box"
            elif self.__mesh.type == MESH_2D_TORUS_DIAGONAL:
                mesh = "Diagonal Torus"
            elif self.__mesh.type == MESH_2D_TORUS_NATURAL:
                mesh = "Natural Torus"

            if self.__num_particles == 1:
                particles = str(self.__num_particles) + " Particle on a "
            else:
                particles = str(self.__num_particles) + " Particles on a "

            return "Quantum Walk with " + particles + mesh

    def output_filename(self):
        if self.__mesh.is_1d():
            if self.__mesh.type == MESH_1D_LINE:
                mesh = "LINE"
            elif self.__mesh.type == MESH_1D_SEGMENT:
                mesh = "SEGMENT"
            elif self.__mesh.type == MESH_1D_CYCLE:
                mesh = "CYCLE"

            size = str(self.__mesh.size)

            return "DTQW1D_{}_{}_{}_{}_{}".format(
                mesh, size, self.__steps, self.__num_particles, self.__min_partitions
            )
        elif self.__mesh.is_2d():
            if self.__mesh.type == MESH_2D_LATTICE_DIAGONAL:
                mesh = "LATTICE_DIAGONAL"
            elif self.__mesh.type == MESH_2D_LATTICE_NATURAL:
                mesh = "LATTICE_NATURAL"
            elif self.__mesh.type == MESH_2D_BOX_DIAGONAL:
                mesh = "BOX_DIAGONAL"
            elif self.__mesh.type == MESH_2D_BOX_NATURAL:
                mesh = "BOX_NATURAL"
            elif self.__mesh.type == MESH_2D_TORUS_DIAGONAL:
                mesh = "TORUS_DIAGONAL"
            elif self.__mesh.type == MESH_2D_TORUS_NATURAL:
                mesh = "TORUS_NATURAL"

            size = str(self.__mesh.size[0]) + "-" + str(self.__mesh.size[1])

            return "DTQW2D_{}_{}_{}_{}_{}".format(
                mesh, size, self.__steps, self.__num_particles, self.__min_partitions
            )

    def clear_operators(self):
        if self.__coin_operator is not None:
            self.__coin_operator.destroy()

        if self.__shift_operator is not None:
            self.__shift_operator.destroy()

        if self.__unitary_operator is not None:
            self.__unitary_operator.destroy()

        if self.__interaction_operator is not None:
            self.__interaction_operator.destroy()

        if self.__multiparticles_unitary_operator is not None:
            self.__multiparticles_unitary_operator.destroy()

        if self.__walk_operator is not None:
            self.__walk_operator.destroy()

    def walk(self, steps, initial_state, collision_phase=None):
        if not is_state(initial_state):
            raise TypeError('State instance expected (not "{}"'.format(type(initial_state)))

        if self.__mesh.is_1d():
            if self.__mesh.type == MESH_1D_LINE:
                if steps > int((self.__mesh.size - 1) / 2):
                    raise ValueError("the number of steps cannot be greater than the size of the lattice")
        elif self.__mesh.is_2d():
            if self.__mesh.type == MESH_2D_LATTICE_DIAGONAL or self.__mesh.type == MESH_2D_LATTICE_NATURAL:
                if steps > int((self.__mesh.size[0] - 1) / 2) or steps > int((self.__mesh.size[1] - 1) / 2):
                    raise ValueError("the number of steps cannot be greater than the size of the lattice")

        self.__steps = steps

        self.__logger.info("Steps: {}".format(self.__steps))
        self.__logger.info("Space size: {}".format(self.__mesh.size))
        self.__logger.info("Nº of partitions: {}".format(self.__min_partitions))
        self.__logger.info("Nº of particles: {}".format(self.__num_particles))
        if self.__num_particles > 1:
            self.__logger.info("Collision phase: {}".format(collision_phase))

        result = initial_state

        if self.__steps > 0:
            result = result.to_rdd(self.__min_partitions, True)
            result.materialize()

            if not result.is_unitary():
                raise ValueError("the initial state is not unitary")

            self.__logger.info("Initial state is consuming {} bytes".format(result.memory_usage))
            self.__logger.debug("Initial state format: {}".format(result.format))
            if result.is_path():
                self.__logger.info("Initial state path: {}".format(result.data))
            self.__logger.debug("Shape of initial state: {}".format(result.shape))

            wo = self.__walk_operator = self.create_walk_operator(collision_phase)

            self.__logger.info("Starting the walk...")

            t1 = datetime.now()

            for i in range(self.__steps):
                t_tmp = datetime.now()

                result_tmp = wo.multiply(result, self.__min_partitions)
                result_tmp.clear_rdd_path()
                result.destroy()
                result = result_tmp

                self.__logger.debug("Step {} was done in {}s".format(i + 1, (datetime.now() - t_tmp).total_seconds()))

                t_tmp = datetime.now()

                self.__logger.debug("Checking if the state is unitary...")
                if not result.is_unitary():
                    raise ValueError("the state is not unitary!")

                self.__logger.debug("Unitarity check was done in {}s".format((datetime.now() - t_tmp).total_seconds()))

            t2 = datetime.now()
            self.__execution_times['walk'] = (t2 - t1).total_seconds()
            self.__memory_usage['state'] = result.memory_usage

            self.__logger.info("Walk was done in {}s".format(self.__execution_times['walk']))
            self.__logger.info("Final state is consuming {} bytes".format(self.__memory_usage['state']))
            self.__logger.debug("Final state format: {}".format(result.format))
            if result.is_path():
                self.__logger.info("Final state path: {}".format(result))
            self.__logger.debug("Shape of final state: {}".format(result.shape))

            app_id = self.__spark_context.applicationId
            self.__metrics.log_rdds(app_id=app_id)

        return result

    def export_times(self, extension='csv', path=None):
        if extension == 'csv':
            if path is None:
                path = './'
            else:
                create_dir(path)

            f = path + self.output_filename() + "_TIMES." + extension

            fieldnames = self.__execution_times.keys()

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                f.write(','.join(fieldnames) + "\n")
                w.writerow(self.__execution_times)
        elif extension == 'txt':
            str_times = [
                "Coin operator in {}s".format(self.__execution_times['coin_operator']),
                "Shift operator in {}s".format(self.__execution_times['shift_operator']),
                "Unitary operator in {}s".format(self.__execution_times['unitary_operator']),
                "Interaction operator in {}s".format(self.__execution_times['interaction_operator']),
                "Multiparticles unitary operator in {}s".format(
                    self.__execution_times['multiparticles_unitary_operator']
                ),
                "Walk operator in {}s".format(self.__execution_times['walk_operator']),
                "Walk in {}s".format(self.__execution_times['walk']),
                "Plots in {}s".format(self.__execution_times['export_plot'])
            ]

            str_times = '\n'.join(str_times)

            if path is None:
                print(str_times)
            else:
                create_dir(path)

                f = path + self.output_filename() + "_TIMES." + extension

                with open(f, 'w') as f:
                    f.write(str_times)
        else:
            raise Exception("Unsupported file extension!")

    def export_memory(self, extension='csv', path=None):
        if extension == 'csv':
            if path is None:
                path = './'
            else:
                create_dir(path)

            f = path + self.output_filename() + "_MEMORY." + extension

            fieldnames = self.__memory_usage.keys()

            with open(f, 'w') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                f.write(','.join(fieldnames) + "\n")
                w.writerow(self.__memory_usage)
        elif extension == 'txt':
            str_memory = [
                "Coin operator is consuming {} bytes".format(self.__memory_usage['coin_operator']),
                "Shift operator is consuming {} bytes".format(self.__memory_usage['shift_operator']),
                "Unitary operator is consuming {} bytes".format(self.__memory_usage['unitary_operator']),
                "Interaction operator is consuming {} bytes".format(self.__memory_usage['interaction_operator']),
                "Multiparticles unitary operator is consuming {} bytes".format(
                    self.__memory_usage['unitary_operator']
                ),
                "Walk operator is consuming {} bytes".format(self.__memory_usage['walk_operator']),
                "State is consuming {} bytes".format(self.__memory_usage['state'])
            ]

            str_memory = '\n'.join(str_memory)

            if path is None:
                print(str_memory)
            else:
                create_dir(path)

                f = path + self.output_filename() + "_MEMORY." + extension

                with open(f, 'w') as f:
                    f.write(str_memory)
        else:
            raise Exception("Unsupported file extension!")

    @staticmethod
    def __build_onedim_plot(pdf, axis, labels, title):
        plt.cla()
        plt.clf()

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

    @staticmethod
    def __build_twodim_plot(pdf, axis, labels, title):
        plt.cla()
        plt.clf()

        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

        surface = axes.plot_surface(
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

    def export_plots(self, pdfs, path=None, **kwargs):
        self.__logger.info("Start ploting probabilities...")
        t1 = datetime.now()

        for k, v in pdfs.items():
            title = self.plot_title()
            onedim = False
            twodim = False

            if self.__mesh.is_1d():
                if self.__mesh.type == MESH_1D_LINE:
                    axis = range(- int((self.__mesh.size - 1) / 2), int((self.__mesh.size - 1) / 2) + 1)
                else:
                    axis = range(self.__mesh.size)

                labels = "Position", "Probability"
                shape = (self.__mesh.size, 1)
            elif self.__mesh.is_2d():
                if self.__mesh.type == MESH_2D_LATTICE_DIAGONAL or self.__mesh.type == MESH_2D_LATTICE_NATURAL:
                    axis = np.meshgrid(
                        range(- int((self.__mesh.size[0] - 1) / 2), int((self.__mesh.size[0] - 1) / 2) + 1),
                        range(- int((self.__mesh.size[1] - 1) / 2), int((self.__mesh.size[1] - 1) / 2) + 1)
                    )
                else:
                    axis = np.meshgrid(range(self.__mesh.size[0]), range(self.__mesh.size[1]))

                labels = "Position X", "Position Y", "Probability"
                shape = (self.__mesh.size[0], self.__mesh.size[1])

            if k == 'full_measurement':
                if self.__mesh.is_1d():
                    if self.__num_particles > 2:
                        continue

                    pdf = v
                    plot_title = title

                    if self.__num_particles == 2:
                        if self.__mesh.type == MESH_1D_LINE:
                            axis = np.meshgrid(
                                range(- int((self.__mesh.size - 1) / 2), int((self.__mesh.size - 1) / 2) + 1),
                                range(- int((self.__mesh.size - 1) / 2), int((self.__mesh.size - 1) / 2) + 1)
                            )
                        else:
                            axis = np.meshgrid(range(self.__mesh.size), range(self.__mesh.size))

                        labels = "Position X1", "Position X2", "Probability"
                        shape = (self.__mesh.size, self.__mesh.size)

                        twodim = True
                    else:
                        onedim = True
                elif self.__mesh.is_2d:
                    if self.__num_particles > 1:
                        continue

                    pdf = v
                    plot_title = title
                    twodim = True

                if path is not None:
                    filename = path + self.output_filename() + "_" + k.upper() + ".png"
            elif k == 'filtered_measurement':
                pdf = v
                plot_title = title

                if self.__mesh.is_1d():
                    onedim = True
                elif self.__mesh.is_2d():
                    twodim = True

                if path is not None:
                    filename = path + self.output_filename() + "_" + k.upper() + ".png"
            elif k == 'partial_measurement':
                for i in range(len(v)):
                    pdf = v[i]
                    plot_title = title + " (Particle " + str(i + 1) + ")"

                    if self.__mesh.is_1d():
                        onedim = True
                    elif self.__mesh.is_2d():
                        twodim = True

                    if path is not None:
                        filename = path + self.output_filename() + "_" + k.upper() + "_" + str(i + 1) + ".png"

                    pdf = pdf.to_dense(True)

                    if onedim:
                        self.__build_onedim_plot(pdf.data, axis, labels, plot_title)
                    elif twodim:
                        self.__build_twodim_plot(pdf.data, axis, labels, plot_title)

                    if path is None:
                        plt.show()
                    else:
                        plt.savefig(filename, kwargs=kwargs)

                continue

            pdf = pdf.to_dense(True)

            if onedim:
                self.__build_onedim_plot(pdf.data, axis, labels, plot_title)
            elif twodim:
                self.__build_twodim_plot(pdf.data, axis, labels, plot_title)

            if path is None:
                plt.show()
            else:
                plt.savefig(filename, kwargs=kwargs)

        t2 = datetime.now()
        self.__execution_times['export_plot'] = (t2 - t1).total_seconds()

        self.__logger.info("Plots in {}s".format(self.__execution_times['export_plot']))
