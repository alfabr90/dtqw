import cmath
from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.utils.logger import Logger
from dtqw.utils.metrics import Metrics
from dtqw.math.operator import Operator, is_operator
from dtqw.math.state import State


class DiscreteTimeQuantumWalk:
    def __init__(self, spark_context, coin, mesh, num_particles, log_filename='./log.txt'):
        self._spark_context = spark_context
        self._coin = coin
        self._mesh = mesh
        self._num_particles = num_particles
        self._min_partitions = 8

        self._coin_operator = None
        self._shift_operator = None
        self._unitary_operator = None
        self._interaction_operator = None
        self._multiparticle_unitary_operator = None
        self._walk_operator = None

        self._steps = 0

        self._logger = Logger(__name__, log_filename)
        self._metrics = Metrics(log_filename=log_filename)

        self._execution_times = {
            'coin_operator': 0.0,
            'shift_operator': 0.0,
            'unitary_operator': 0.0,
            'interaction_operator': 0.0,
            'multiparticle_unitary_operator': 0.0,
            'walk_operator': 0.0,
            'walk': 0.0,
            'export_plot': 0.0
        }

        self._memory_usage = {
            'coin_operator': 0,
            'shift_operator': 0,
            'unitary_operator': 0,
            'interaction_operator': 0,
            'multiparticle_unitary_operator': 0,
            'walk_operator': 0,
            'state': 0
        }

        if num_particles < 1:
            self._logger.error("Invalid number of particles")
            raise ValueError("invalid number of particles")

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def coin(self):
        return self._coin

    @property
    def mesh(self):
        return self._mesh

    @property
    def memory_usage(self):
        return self._memory_usage

    @property
    def execution_times(self):
        return self._execution_times

    @property
    def coin_operator(self):
        return self._coin_operator

    @property
    def shift_operator(self):
        return self._shift_operator

    @property
    def unitary_operator(self):
        return self._unitary_operator

    @property
    def interaction_operator(self):
        return self._interaction_operator

    @property
    def multiparticle_unitary_operator(self):
        return self._multiparticle_unitary_operator

    @property
    def walk_operator(self):
        return self._walk_operator

    @coin_operator.setter
    def coin_operator(self, co):
        if is_operator(co):
            self._coin_operator = co
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(co)))
            raise TypeError('operator instance expected, not "{}"'.format(type(co)))

    @shift_operator.setter
    def shift_operator(self, so):
        if is_operator(so):
            self._shift_operator = so
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(so)))
            raise TypeError('operator instance expected, not "{}"'.format(type(so)))

    @unitary_operator.setter
    def unitary_operator(self, uo):
        if is_operator(uo):
            self._unitary_operator = uo
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(uo)))
            raise TypeError('operator instance expected, not "{}"'.format(type(uo)))

    @interaction_operator.setter
    def interaction_operator(self, io):
        if is_operator(io):
            self._interaction_operator = io
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(io)))
            raise TypeError('operator instance expected, not "{}"'.format(type(io)))

    @multiparticle_unitary_operator.setter
    def multiparticle_unitary_operator(self, mu):
        if is_operator(mu):
            self._multiparticle_unitary_operator = mu
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(mu)))
            raise TypeError('operator instance expected, not "{}"'.format(type(mu)))

    @walk_operator.setter
    def walk_operator(self, wo):
        if is_operator(wo):
            self._walk_operator = wo
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(wo)))
            raise TypeError('operator instance expected, not "{}"'.format(type(wo)))

    def create_coin_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Building coin operator...")
        t1 = datetime.now()

        self._coin_operator = self._coin.create_operator(self._mesh, storage_level).materialize(storage_level)

        self._execution_times['coin_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Coin operator was built in {}s".format(self._execution_times['coin_operator']))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_shift_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Building shift operator...")
        t1 = datetime.now()

        self._shift_operator = self._mesh.create_operator(storage_level).materialize(storage_level)

        self._execution_times['shift_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Shift operator was built in {}s".format(self._execution_times['shift_operator']))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_unitary_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Building unitary operator...")

        if self._coin_operator is None:
            self._logger.info("No coin operator has been set. A new one will be built")
            self.create_coin_operator(storage_level)

        if self._shift_operator is None:
            self._logger.info("No shift operator has been set. A new one will be built")
            self.create_shift_operator(storage_level)

        t1 = datetime.now()

        num_partitions = max(self._shift_operator.data.getNumPartitions(), self._coin_operator.data.getNumPartitions())

        rdd = self._shift_operator.data.map(
            lambda m: (m[1], (m[0], m[2]))
        ).partitionBy(
            numPartitions=num_partitions
        )

        so = Operator(self._spark_context, rdd, self._shift_operator.shape, self._logger.filename)

        rdd = self._coin_operator.data.map(
            lambda m: (m[0], (m[1], m[2]))
        ).partitionBy(
            numPartitions=num_partitions
        )

        co = Operator(self._spark_context, rdd, self._shift_operator.shape, self._logger.filename)

        self._unitary_operator = so.multiply(co).materialize(storage_level)

        self._coin_operator.unpersist()
        self._shift_operator.unpersist()

        self._execution_times['unitary_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Unitary operator was built in {}s".format(self._execution_times['unitary_operator']))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_interaction_operator(self, phase, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Building interaction operator...")
        t1 = datetime.now()

        phase = cmath.exp(phase * (0.0 + 1.0j))
        num_particles = self._num_particles

        coin_size = 2

        if self._mesh.is_1d():
            size = self._mesh.size
            cs_size = coin_size * self._mesh.size

            shape = (cs_size, cs_size)
            rdd_range = cs_size ** num_particles

            def __map(m):
                a = []

                for p in range(num_particles):
                    a.append(int(m / (cs_size ** (num_particles - 1 - p))) % size)

                for i in range(num_particles):
                    if a[0] != a[i]:
                        return m, m, 1

                return m, m, phase
        elif self._mesh.is_2d():
            ndim = 2
            ind = ndim * num_particles

            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            cs_size_xy = cs_size_x * cs_size_y

            shape = (cs_size_xy, cs_size_xy)
            rdd_range = cs_size_xy ** num_particles

            def __map(m):
                a = []

                for p in range(num_particles):
                    a.append(int(m / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x)
                    a.append(int(m / (cs_size_xy ** (num_particles - 1 - p))) % size_y)

                for i in range(0, ind, ndim):
                    if a[0] != a[i] or a[1] != a[i + 1]:
                        return m, m, 1

                return m, m, phase
        else:
            self._logger.error("Mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self._spark_context.range(
            rdd_range
        ).map(
            __map
        )

        self._interaction_operator = Operator(
            self._spark_context, rdd, shape, self._logger.filename
        ).materialize(storage_level)

        self._execution_times['interaction_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info(
            "Interaction operator was built in {}s".format(self._execution_times['interaction_operator'])
        )

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_multiparticle_unitary_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Building multiparticle unitary operator...")

        if self._unitary_operator is None:
            self._logger.info("No unitary operator has been set. A new one will be built")
            self.create_unitary_operator(storage_level)

        t1 = datetime.now()

        op_tmp = self._unitary_operator
        uo = broadcast(self._spark_context, self._unitary_operator.data.collect())

        for i in range(self._num_particles - 1):
            op_tmp = op_tmp.kron(uo, self._unitary_operator.shape)

        self._multiparticle_unitary_operator = op_tmp.materialize(storage_level)

        self._execution_times['multiparticle_unitary_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info(
            "Multiparticle unitary operator was built in {}s".format(
                self._execution_times['multiparticle_unitary_operator']
            )
        )

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_walk_operator(self, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        t1 = datetime.now()

        app_id = self._spark_context.applicationId

        if self._num_particles == 1:
            self._logger.info("With just one particle, the walk operator is the unitary operator")

            if self._unitary_operator is None:
                self._logger.info("No unitary operator has been set. A new one will be built")
                self.create_unitary_operator(storage_level)

            num_partitions = self._unitary_operator.data.getNumPartitions()

            rdd = self._unitary_operator.data.map(
                lambda m: (m[1], (m[0], m[2]))
            ).partitionBy(
                numPartitions=num_partitions
            )

            self._walk_operator = Operator(
                self._spark_context, rdd, self._unitary_operator.shape, self._logger.filename
            ).materialize(storage_level)

            self._unitary_operator.unpersist()
        else:
            if phase is None:
                self._logger.info("No collision phase has been defined. "
                                  "The walk operator will be the multiparticle unitary operator")

                if self._multiparticle_unitary_operator is None:
                    self._logger.info("No multiparticle unitary operator has been set. A new one will be built")
                    self.create_multiparticle_unitary_operator(storage_level)

                num_partitions = self._multiparticle_unitary_operator.data.getNumPartitions()

                rdd = self._multiparticle_unitary_operator.data.map(
                    lambda m: (m[1], (m[0], m[2]))
                ).partitionBy(
                    numPartitions=num_partitions
                )

                self._walk_operator = Operator(
                    self._spark_context, rdd, self._unitary_operator.shape, self._logger.filename
                ).materialize(storage_level)

                self._unitary_operator.unpersist()
            else:
                self._logger.info("Building walk operator...")

                if self._multiparticle_unitary_operator is None:
                    self._logger.info("No multiparticle unitary operator has been set. A new one will be built")
                    self.create_multiparticle_unitary_operator(storage_level)

                if self._interaction_operator is None:
                    self._logger.info("No interaction operator has been set. A new one will be built")
                    self.create_interaction_operator(phase, storage_level)

                t1 = datetime.now()

                shape = self._multiparticle_unitary_operator.shape

                num_partitions = max(
                    self._multiparticle_unitary_operator.data.getNumPartitions(),
                    self._interaction_operator.data.getNumPartitions()
                )

                self._logger.info("Multiplying multiparticle unitary operator with interaction operation...")

                rdd = self._multiparticle_unitary_operator.data.map(
                    lambda m: (m[1], (m[0], m[2]))
                ).partitionBy(
                    numPartitions=num_partitions
                )

                muo = Operator(self._spark_context, rdd, shape, self._logger.filename)

                rdd = self._interaction_operator.data.map(
                    lambda m: (m[0], (m[1], m[2]))
                ).partitionBy(
                    numPartitions=num_partitions
                )

                io = Operator(self._spark_context, rdd, shape, self._logger.filename)

                op_tmp = muo.multiply(io)

                rdd = op_tmp.data.map(
                    lambda m: (m[1], (m[0], m[2]))
                ).partitionBy(
                    numPartitions=num_partitions
                )

                self._walk_operator = Operator(
                    self._spark_context, rdd, shape, self._logger.filename
                ).materialize(storage_level)

                self._multiparticle_unitary_operator.unpersist()
                self._interaction_operator.unpersist()

                self._logger.debug(
                    "Multiplication between multiparticle unitary operator and "
                    "interaction operator was done in {}s".format((datetime.now() - t1).total_seconds())
                )

        self._execution_times['walk_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info(
            "Walk operator was built in {}s".format(self._execution_times['walk_operator'])
        )

        self._metrics.log_executors(app_id=app_id)
        self._metrics.log_rdds(app_id=app_id)

    def title(self):
        return "Quantum Walk with {} Particle(s) on a {}".format(self._num_particles, self._mesh.title())

    def filename(self):
        return "{}_{}_{}".format(
            self._mesh.filename(), self._steps, self._num_particles
        )

    def destroy_operators(self):
        if self._coin_operator is not None:
            self._coin_operator.destroy()

        if self._shift_operator is not None:
            self._shift_operator.destroy()

        if self._unitary_operator is not None:
            self._unitary_operator.destroy()

        if self._interaction_operator is not None:
            self._interaction_operator.destroy()

        if self._multiparticle_unitary_operator is not None:
            self._multiparticle_unitary_operator.destroy()

        if self._walk_operator is not None:
            self._walk_operator.destroy()

    def walk(self, steps, initial_state, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        if not self._mesh.check_steps(steps):
            self._logger.error("Invalid number of steps")
            raise ValueError("invalid number of steps")

        self._steps = steps

        self._logger.info("Steps: {}".format(self._steps))
        self._logger.info("Space size: {}".format(self._mesh.size))
        self._logger.info("Nº of partitions: {}".format(self._min_partitions))
        self._logger.info("Nº of particles: {}".format(self._num_particles))
        if self._num_particles > 1:
            self._logger.info("Collision phase: {}".format(phase))

        app_id = self._spark_context.applicationId

        result = initial_state

        if not result.is_unitary():
            self._logger.error("The initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        if self._steps > 0:
            self.create_walk_operator(phase, storage_level)

            wo = self._walk_operator
            self._logger.debug("\n" + wo.data.toDebugString().decode())

            t1 = datetime.now()

            num_partitions = wo.data.getNumPartitions()

            rdd = result.data.partitionBy(
                numPartitions=num_partitions
            )

            result = State(
                self._spark_context, rdd, result.shape, self._mesh, self._num_particles, self._logger.filename
            ).materialize(storage_level)

            self._logger.info("Starting the walk...")

            for i in range(self._steps):
                self._metrics.log_executors(app_id=app_id)
                self._metrics.log_rdds(app_id=app_id)

                t_tmp = datetime.now()

                result_tmp = wo.multiply(result).materialize(storage_level)

                result.unpersist()

                result = result_tmp

                self._logger.debug(
                    "Step {} was done in {}s".format(i + 1, (datetime.now() - t_tmp).total_seconds()))

                t_tmp = datetime.now()

                self._logger.debug("Checking if the state is unitary...")

                if not result.is_unitary():
                    self._logger.error("The state is not unitary")
                    raise ValueError("the state is not unitary")

                self._logger.debug(
                    "Unitarity check was done in {}s".format((datetime.now() - t_tmp).total_seconds()))

            self._execution_times['walk'] = (datetime.now() - t1).total_seconds()

            self._logger.info("Walk was done in {}s".format(self._execution_times['walk']))

        self._metrics.log_executors(app_id=app_id)
        self._metrics.log_rdds(app_id=app_id)

        return result
