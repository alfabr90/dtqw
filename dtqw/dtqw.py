import cmath
from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.utils.logger import Logger
from dtqw.utils.metrics import Metrics
from dtqw.math.operator import Operator, is_operator
from dtqw.math.state import State


class DiscreteTimeQuantumWalk:
    def __init__(self, spark_context, coin, mesh, num_particles, num_partitions, log_filename='./log.txt'):
        self._spark_context = spark_context
        self._coin = coin
        self._mesh = mesh
        self._num_particles = num_particles
        self._num_partitions = num_partitions

        self._coin_operator = None
        self._shift_operator = None
        self._unitary_operator = None
        self._interaction_operator = None
        self._walk_operator = None

        self._steps = 0

        self._logger = Logger(__name__, log_filename)
        self._metrics = Metrics(log_filename=log_filename)

        self._execution_times = {
            'coin_operator': 0.0,
            'shift_operator': 0.0,
            'unitary_operator': 0.0,
            'interaction_operator': 0.0,
            'walk_operator': 0.0,
            'walk': 0.0
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

    @walk_operator.setter
    def walk_operator(self, wo):
        if is_operator(wo):
            self._walk_operator = wo
        elif isinstance(wo, (list, tuple)):
            if len(wo) != self._num_particles:
                self._logger.error('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))
                raise ValueError('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))

            for o in wo:
                if not is_operator(o):
                    self._logger.error('Operator instance expected, not "{}"'.format(type(wo)))
                    raise TypeError('operator instance expected, not "{}"'.format(type(wo)))
        else:
            self._logger.error('Operator instance expected, not "{}"'.format(type(wo)))
            raise TypeError('operator instance expected, not "{}"'.format(type(wo)))

    def create_coin_operator(self):
        self._logger.info("Building coin operator...")
        t1 = datetime.now()

        self._coin_operator = self._coin.create_operator(self._mesh).dump()

        self._execution_times['coin_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Coin operator was built in {}s".format(self._execution_times['coin_operator']))
        self._logger.debug("Shape of coin operator: {}".format(self._coin_operator.shape))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_shift_operator(self):
        self._logger.info("Building shift operator...")
        t1 = datetime.now()

        self._shift_operator = self._mesh.create_operator().dump()

        self._execution_times['shift_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Shift operator was built in {}s".format(self._execution_times['shift_operator']))
        self._logger.debug("Shape of shift operator: {}".format(self._shift_operator.shape))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_unitary_operator(self):
        self._logger.info("Building unitary operator...")

        if self._coin_operator is None:
            self._logger.info("No coin operator has been set. A new one will be built")
            self.create_coin_operator()

        if self._shift_operator is None:
            self._logger.info("No shift operator has been set. A new one will be built")
            self.create_shift_operator()

        t1 = datetime.now()

        rdd = self._shift_operator.data.map(
            lambda m: (m[1], (m[0], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        so = Operator(self._spark_context, rdd, self._shift_operator.shape, self._logger.filename)

        rdd = self._coin_operator.data.map(
            lambda m: (m[0], (m[1], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        co = Operator(self._spark_context, rdd, self._shift_operator.shape, self._logger.filename)

        self._unitary_operator = so.multiply(co).dump()

        self._coin_operator.unpersist()
        self._shift_operator.unpersist()

        self._execution_times['unitary_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info("Unitary operator was built in {}s".format(self._execution_times['unitary_operator']))
        self._logger.debug("Shape of unitary operator: {}".format(self._unitary_operator.shape))

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

            rdd_range = cs_size ** num_particles
            shape = (rdd_range, rdd_range)

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

            rdd_range = cs_size_xy ** num_particles
            shape = (rdd_range, rdd_range)

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

        io = Operator(
            self._spark_context, rdd, shape, self._logger.filename
        ).dump()

        rdd = io.data.map(
            lambda m: (m[0], (m[1], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        self._interaction_operator = Operator(
            self._spark_context, rdd, io.shape, self._logger.filename
        ).materialize(storage_level)

        self._execution_times['interaction_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info(
            "Interaction operator was built in {}s".format(self._execution_times['interaction_operator'])
        )
        self._logger.debug("Shape of interaction operator: {}".format(self._interaction_operator.shape))

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def create_walk_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        t1 = datetime.now()

        if self._unitary_operator is None:
            self._logger.info("No unitary operator has been set. A new one will be built")
            self.create_unitary_operator()

        if self._num_particles == 1:
            self._logger.info("With just one particle, the walk operator is the unitary operator")

            rdd = self._unitary_operator.data.map(
                lambda m: (m[0], (m[1], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )

            self._walk_operator = Operator(
                self._spark_context, rdd, self._unitary_operator.shape, self._logger.filename
            ).materialize(storage_level)
        else:
            self._logger.info("Building walk operator...")

            t1 = datetime.now()

            shape = self._unitary_operator.shape

            rdd = self._spark_context.range(
                shape[0]
            ).map(
                lambda m: (m, m, 1)
            )

            identity = Operator(self._spark_context, rdd, shape, self._logger.filename).materialize(storage_level)
            io = broadcast(self._spark_context, identity.data.collect())
            uo = broadcast(self._spark_context, self._unitary_operator.data.collect())

            self._walk_operator = []

            for p in range(self._num_particles):
                self._logger.debug("Building walk operator for particle {}...".format(p + 1))

                t_tmp = datetime.now()

                if p == 0:
                    op_tmp = self._unitary_operator

                    for i in range(self._num_particles - 1 - p):
                        op_tmp = op_tmp.kron(io, identity.shape)
                else:
                    op_tmp = identity

                    for i in range(p - 1):
                        op_tmp = op_tmp.kron(io, identity.shape)

                    op_tmp = op_tmp.kron(uo, self._unitary_operator.shape)

                    for i in range(self._num_particles - 1 - p):
                        op_tmp = op_tmp.kron(io, identity.shape)

                op_tmp = op_tmp.dump()

                rdd = op_tmp.data.map(
                    lambda m: (m[0], (m[1], m[2]))
                ).partitionBy(
                    numPartitions=self._num_partitions
                )

                self._walk_operator.append(
                    Operator(
                        self._spark_context, rdd, op_tmp.shape, self._logger.filename
                    ).materialize(storage_level)
                )

                self._logger.debug(
                    "Walk operator for particle {} was built in {}s".format(
                        p + 1, (datetime.now() - t_tmp).total_seconds()
                    )
                )

            uo.unpersist()
            io.unpersist()
            identity.destroy()

        self._execution_times['walk_operator'] = (datetime.now() - t1).total_seconds()

        self._logger.info(
            "Walk operator was built in {}s".format(self._execution_times['walk_operator'])
        )

        if self._num_particles == 1:
            self._logger.debug("Shape of walk operator: {}".format(self._walk_operator.shape))
        else:
            for o in range(len(self._walk_operator)):
                self._logger.debug(
                    "Shape of walk operator for particle {}: {}".format(o + 1, self._walk_operator[o].shape)
                )

        app_id = self._spark_context.applicationId
        self._metrics.log_rdds(app_id=app_id)

    def title(self):
        return "Quantum Walk with {} Particle(s) on a {}".format(self._num_particles, self._mesh.title())

    def filename(self):
        return "{}_{}_{}".format(self._mesh.filename(), self._steps, self._num_particles)

    def destroy_operators(self):
        if self._coin_operator is not None:
            self._coin_operator.destroy()

        if self._shift_operator is not None:
            self._shift_operator.destroy()

        if self._unitary_operator is not None:
            self._unitary_operator.destroy()

        if self._interaction_operator is not None:
            self._interaction_operator.destroy()

        if self._num_particles == 1:
            if self._walk_operator is not None:
                self._walk_operator.destroy()
        else:
            for wo in self._walk_operator:
                if wo is not None:
                    wo.destroy()

    def _monoparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Starting the walk...")

        wo = self._walk_operator

        self._logger.debug("Walk operator lineage:\n" + wo.data.toDebugString().decode())

        result = initial_state

        for i in range(steps):
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

        return result

    def _multiparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        self._logger.info("Starting the walk...")

        wo = self._walk_operator
        io = self._interaction_operator

        for o in range(len(wo)):
            self._logger.debug(
                "Walk operator lineage for particle {}:\n".format(o + 1) + wo[o].data.toDebugString().decode()
            )

        result = initial_state

        for i in range(steps):
            t_tmp = datetime.now()

            if io is not None:
                result_tmp = io.multiply(result)

                for o in wo:
                    result_tmp = o.multiply(result_tmp)
            else:
                result_tmp = wo[0].multiply(result)

                for o in range(len(wo) - 1):
                    result_tmp = wo[o].multiply(result_tmp)

            result_tmp.materialize(storage_level)
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

        return result

    def walk(self, steps, initial_state, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        if not self._mesh.check_steps(steps):
            self._logger.error("Invalid number of steps")
            raise ValueError("invalid number of steps")

        self._steps = steps

        self._logger.info("Steps: {}".format(self._steps))
        self._logger.info("Space size: {}".format(self._mesh.size))
        self._logger.info("Nº of particles: {}".format(self._num_particles))
        if self._num_particles > 1:
            self._logger.info("Collision phase: {}".format(phase))
        self._logger.info("Nº of partitions: {}".format(self._num_partitions))

        app_id = self._spark_context.applicationId

        rdd = initial_state.data.partitionBy(
            numPartitions=self._num_partitions
        )

        result = State(
            self._spark_context, rdd, initial_state.shape, self._mesh, self._num_particles, self._logger.filename
        ).materialize(storage_level)

        initial_state.unpersist()

        self._logger.debug("Shape of initial state: {}".format(result.shape))

        if not result.is_unitary():
            self._logger.error("The initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        self._metrics.log_executors(app_id=app_id)
        self._metrics.log_rdds(app_id=app_id)

        if self._steps > 0:
            if self._walk_operator is None:
                self._logger.info("No walk operator has been set. A new one will be built")
                self.create_walk_operator(storage_level)

            if self._num_particles > 1:
                if phase is None:
                    self._logger.info("No collision phase has been defined")
                else:
                    if self._interaction_operator is None:
                        self._logger.info("No interaction operator has been set. A new one will be built")
                        self.create_interaction_operator(phase, storage_level)

            t1 = datetime.now()

            if self._num_particles == 1:
                result = self._monoparticle_walk(steps, result, storage_level)
            else:
                result = self._multiparticle_walk(steps, result, storage_level)

            self._execution_times['walk'] = (datetime.now() - t1).total_seconds()

            self._logger.info("Walk was done in {}s".format(self._execution_times['walk']))

        self._metrics.log_executors(app_id=app_id)
        self._metrics.log_rdds(app_id=app_id)

        return result
