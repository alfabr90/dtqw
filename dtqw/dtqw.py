import cmath
from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.math.operator import *
from dtqw.math.state import *

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    def __init__(self, spark_context, coin, mesh, num_particles, num_partitions):
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

        if num_particles < 1:
            # self.logger.error("Invalid number of particles")
            raise ValueError("invalid number of particles")

        self.logger = None
        self.profiler = None

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
            if self.logger:
                self.logger.error('Operator instance expected, not "{}"'.format(type(co)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(co)))

    @shift_operator.setter
    def shift_operator(self, so):
        if is_operator(so):
            self._shift_operator = so
        else:
            if self.logger:
                self.logger.error('Operator instance expected, not "{}"'.format(type(so)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(so)))

    @unitary_operator.setter
    def unitary_operator(self, uo):
        if is_operator(uo):
            self._unitary_operator = uo
        else:
            if self.logger:
                self.logger.error('Operator instance expected, not "{}"'.format(type(uo)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(uo)))

    @interaction_operator.setter
    def interaction_operator(self, io):
        if is_operator(io):
            self._interaction_operator = io
        else:
            if self.logger:
                self.logger.error('Operator instance expected, not "{}"'.format(type(io)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(io)))

    @walk_operator.setter
    def walk_operator(self, wo):
        if is_operator(wo):
            self._walk_operator = wo
        elif isinstance(wo, (list, tuple)):
            if len(wo) != self._num_particles:
                if self.logger:
                    self.logger.error('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))
                raise ValueError('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))

            for o in wo:
                if not is_operator(o):
                    if self.logger:
                        self.logger.error('Operator instance expected, not "{}"'.format(type(wo)))
                    raise TypeError('Operator instance expected, not "{}"'.format(type(wo)))
        else:
            if self.logger:
                self.logger.error('Operator instance expected, not "{}"'.format(type(wo)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(wo)))

    def __str__(self):
        return self.__class__.__name__

    def to_string(self):
        return self.__str__()

    def create_coin_operator(self):
        if self.logger:
            self.logger.info("building coin operator...")
        t1 = datetime.now()

        self._coin_operator = self._coin.create_operator(self._mesh).dump()

        app_id = self._spark_context.applicationId
        rdd_id = self._coin_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('coinOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_rdd('coinOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info("coin operator was built in {}s".format(self.profiler.get_time('coinOperator')))
                self.logger.info(
                    "coin operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd('coinOperator', 'memoryUsed'),
                        self.profiler.get_rdd('coinOperator', 'diskUsed')
                    )
                )
                self.logger.debug("shape of coin operator: {}".format(self._coin_operator.shape))

            self.profiler.log_rdd(app_id=app_id)

    def create_shift_operator(self):
        if self.logger:
            self.logger.info("building shift operator...")
        t1 = datetime.now()

        self._shift_operator = self._mesh.create_operator().dump()

        app_id = self._spark_context.applicationId
        rdd_id = self._shift_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('shiftOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_rdd('shiftOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "shift operator was built in {}s".format(self.profiler.get_time('shiftOperator'))
                )
                self.logger.info(
                    "shift operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd('shiftOperator', 'memoryUsed'),
                        self.profiler.get_rdd('shiftOperator', 'diskUsed')
                    )
                )
                self.logger.debug("shape of shift operator: {}".format(self._shift_operator.shape))

            self.profiler.log_rdd(app_id=app_id)

    def create_unitary_operator(self):
        if self.logger:
            self.logger.info("building unitary operator...")

        if self._coin_operator is None:
            if self.logger:
                self.logger.info("no coin operator has been set. A new one will be built")
            self.create_coin_operator()

        if self._shift_operator is None:
            if self.logger:
                self.logger.info("no shift operator has been set. A new one will be built")
            self.create_shift_operator()

        t1 = datetime.now()

        rdd = self._shift_operator.data.map(
            lambda m: (m[1], (m[0], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        so = Operator(self._spark_context, rdd, self._shift_operator.shape)

        rdd = self._coin_operator.data.map(
            lambda m: (m[0], (m[1], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        co = Operator(self._spark_context, rdd, self._shift_operator.shape)

        self._unitary_operator = so.multiply(co).dump()

        self._coin_operator.unpersist()
        self._shift_operator.unpersist()

        app_id = self._spark_context.applicationId
        rdd_id = self._unitary_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('unitaryOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_rdd('unitaryOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "unitary operator was built in {}s".format(self.profiler.get_time('unitaryOperator'))
                )
                self.logger.info(
                    "unitary operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd('unitaryOperator', 'memoryUsed'),
                        self.profiler.get_rdd('unitaryOperator', 'diskUsed')
                    )
                )
                self.logger.debug("shape of unitary operator: {}".format(self._unitary_operator.shape))

            self.profiler.log_rdd(app_id=app_id)

    def create_interaction_operator(self, phase, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.logger:
            self.logger.info("building interaction operator...")

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
            if self.logger:
                self.logger.error("mesh dimension not implemented")
            raise NotImplementedError("mesh dimension not implemented")

        rdd = self._spark_context.range(
            rdd_range
        ).map(
            __map
        )

        io = Operator(self._spark_context, rdd, shape).dump()

        rdd = io.data.map(
            lambda m: (m[0], (m[1], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        self._interaction_operator = Operator(self._spark_context, rdd, io.shape).materialize(storage_level)

        app_id = self._spark_context.applicationId
        rdd_id = self._interaction_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('interactionOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_rdd('interactionOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "interaction operator was built in {}s".format(self.profiler.get_time('interactionOperator'))
                )
                self.logger.info(
                    "interaction operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd('interactionOperator', 'memoryUsed'),
                        self.profiler.get_rdd('interactionOperator', 'diskUsed')
                    )
                )
                self.logger.debug("shape of interaction operator: {}".format(self._interaction_operator.shape))

            self.profiler.log_rdd(app_id=app_id)

    def create_walk_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self._unitary_operator is None:
            if self.logger:
                self.logger.info("no unitary operator has been set. A new one will be built")
            self.create_unitary_operator()

        t1 = datetime.now()

        if self._num_particles == 1:
            if self.logger:
                self.logger.info("with just one particle, the walk operator is the unitary operator")

            rdd = self._unitary_operator.data.map(
                lambda m: (m[0], (m[1], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )

            self._walk_operator = Operator(
                self._spark_context, rdd, self._unitary_operator.shape
            ).materialize(storage_level)

            app_id = self._spark_context.applicationId
            rdd_id = self._walk_operator.data.id()

            if self.profiler:
                self.profiler.profile_times('walkOperator', (datetime.now() - t1).total_seconds())
                self.profiler.profile_rdd('walkOperator', app_id, rdd_id)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "walk operator was built in {}s".format(self.profiler.get_time('walkOperator'))
                    )
                    self.logger.info(
                        "walk operator is consuming {} bytes in memory and {} bytes in disk".format(
                            self.profiler.get_rdd('walkOperator', 'memoryUsed'),
                            self.profiler.get_rdd('walkOperator', 'diskUsed')
                        )
                    )
                    self.logger.debug("shape of walk operator: {}".format(self._walk_operator.shape))

                self.profiler.log_rdd(app_id=app_id)
        else:
            if self.logger:
                self.logger.info("building walk operator...")

            shape = self._unitary_operator.shape

            rdd = self._spark_context.range(
                shape[0]
            ).map(
                lambda m: (m, m, 1)
            )

            identity = Operator(self._spark_context, rdd, shape).materialize(storage_level)
            io = broadcast(self._spark_context, identity.data.collect())
            uo = broadcast(self._spark_context, self._unitary_operator.data.collect())

            self._walk_operator = []

            for p in range(self._num_particles):
                if self.logger:
                    self.logger.debug("building walk operator for particle {}...".format(p + 1))

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
                    Operator(self._spark_context, rdd, op_tmp.shape).materialize(storage_level)
                )

                if self.logger:
                    self.logger.debug(
                        "walk operator for particle {} was built in {}s".format(
                            p + 1, (datetime.now() - t_tmp).total_seconds()
                        )
                    )

            uo.unpersist()
            io.unpersist()
            identity.destroy()
            self._unitary_operator.unpersist()

            app_id = self._spark_context.applicationId
            rdd_id = [wo.data.id() for wo in self._walk_operator]

            if self.profiler:
                self.profiler.profile_times('walkOperator', (datetime.now() - t1).total_seconds())
                self.profiler.profile_rdd('walkOperator', app_id, rdd_id)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "walk operator was built in {}s".format(self.profiler.get_time('walkOperator'))
                    )
                    self.logger.info(
                        "walk operator is consuming {} bytes in memory and {} bytes in disk".format(
                            self.profiler.get_rdd('walkOperator', 'memoryUsed'),
                            self.profiler.get_rdd('walkOperator', 'diskUsed')
                        )
                    )
                    self.logger.debug("shape of walk operator: {}".format(self._walk_operator[0].shape))

                self.profiler.log_rdd(app_id=app_id)

    def title(self):
        return "Quantum Walk with {} Particle(s) on a {}".format(self._num_particles, self._mesh.title())

    def destroy_operators(self):
        if self.logger:
            self.logger.info('destroying operators...')

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
        wo = self._walk_operator

        app_id = self._spark_context.applicationId

        if self.logger:
            self.logger.debug("walk operator lineage:\n{}".format(wo.data.toDebugString().decode()))

        result = initial_state

        if self.logger:
            self.logger.info("starting the walk...")

        for i in range(steps):
            t_tmp = datetime.now()

            result_tmp = wo.multiply(result).materialize(storage_level)

            result.unpersist()

            result = result_tmp

            if self.logger:
                self.logger.debug("step {} was done in {}s".format(i + 1, (datetime.now() - t_tmp).total_seconds()))

            t_tmp = datetime.now()

            if self.logger:
                self.logger.debug("checking if the state is unitary...")

            if not result.is_unitary():
                if self.logger:
                    self.logger.error("the state is not unitary")
                raise ValueError("the state is not unitary")

            if self.logger:
                self.logger.debug("unitarity check was done in {}s".format((datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self.profiler:
                self.profiler.profile_rdd('systemState{}'.format(i + 1), app_id, rdd_id)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "system state is consuming {} bytes in memory and {} bytes in disk".format(
                            self.profiler.get_rdd('systemState{}'.format(i + 1), 'memoryUsed'),
                            self.profiler.get_rdd('systemState{}'.format(i + 1), 'diskUsed')
                        )
                    )
                    self.logger.debug("shape of initial state: {}".format(result.shape))

        return result

    def _multiparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        wo = self._walk_operator
        io = self._interaction_operator

        app_id = self._spark_context.applicationId

        for o in range(len(wo)):
            if self.logger:
                self.logger.debug(
                    "walk operator lineage for particle {}:\n{}".format(o + 1, wo[o].data.toDebugString().decode())
                )

        if self.logger:
            self.logger.debug("interaction operator lineage:\n{}".format(io.data.toDebugString().decode()))

        result = initial_state

        if self.logger:
            self.logger.info("starting the walk...")

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

            if self.logger:
                self.logger.debug("step {} was done in {}s".format(i + 1, (datetime.now() - t_tmp).total_seconds()))

            t_tmp = datetime.now()

            if self.logger:
                self.logger.debug("checking if the state is unitary...")

            if not result.is_unitary():
                if self.logger:
                    self.logger.error("the state is not unitary")
                raise ValueError("the state is not unitary")

            if self.logger:
                self.logger.debug("unitarity check was done in {}s".format((datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self.profiler:
                self.profiler.profile_rdd('systemState{}'.format(i + 1), app_id, rdd_id)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "system state is consuming {} bytes in memory and {} bytes in disk".format(
                            self.profiler.get_rdd('systemState{}'.format(i + 1), 'memoryUsed'),
                            self.profiler.get_rdd('systemState{}'.format(i + 1), 'diskUsed')
                        )
                    )
                    self.logger.debug("shape of initial state: {}".format(result.shape))

        return result

    def walk(self, steps, initial_state, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        if not self._mesh.check_steps(steps):
            if self.logger:
                self.logger.error("invalid number of steps")
            raise ValueError("invalid number of steps")

        if self.logger:
            self.logger.info("steps: {}".format(steps))
            self.logger.info("space size: {}".format(self._mesh.size))
            self.logger.info("nº of particles: {}".format(self._num_particles))
            self.logger.info("nº of partitions: {}".format(self._num_partitions))

            if self._num_particles > 1:
                self.logger.info("collision phase: {}".format(phase))

        rdd = initial_state.data.partitionBy(
            numPartitions=self._num_partitions
        )

        result = State(
            self._spark_context, rdd, initial_state.shape, self._mesh, self._num_particles
        ).materialize(storage_level)

        initial_state.unpersist()

        if self.logger:
            self.logger.debug("shape of initial state: {}".format(result.shape))

        if not result.is_unitary():
            if self.logger:
                self.logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        app_id = self._spark_context.applicationId
        rdd_id = result.data.id()

        if self.profiler:
            self.profiler.profile_rdd('initialState', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd('initialState', 'memoryUsed'),
                        self.profiler.get_rdd('initialState', 'diskUsed')
                    )
                )
                self.logger.debug("shape of initial state: {}".format(result.shape))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

        if steps > 0:
            if self._walk_operator is None:
                if self.logger:
                    self.logger.info("no walk operator has been set. A new one will be built")
                self.create_walk_operator(storage_level)

            if self._num_particles > 1:
                if phase is None:
                    if self.logger:
                        self.logger.info("no collision phase has been defined")
                else:
                    if self._interaction_operator is None:
                        if self.logger:
                            self.logger.info("no interaction operator has been set. A new one will be built")
                        self.create_interaction_operator(phase, storage_level)

            if self.profiler:
                self.profiler.log_executors(app_id=app_id)
                self.profiler.log_rdd(app_id=app_id)

            t1 = datetime.now()

            if self._num_particles == 1:
                result = self._monoparticle_walk(steps, result, storage_level)
            else:
                result = self._multiparticle_walk(steps, result, storage_level)

            if self.profiler:
                self.profiler.profile_times('walk', (datetime.now() - t1).total_seconds())

                if self.logger:
                    self.logger.info("walk was done in {}s".format(self.profiler.get_time('walk')))

        if self.profiler:
            result.profiler = self.profiler

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

        return result
