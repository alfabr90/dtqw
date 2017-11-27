import cmath
from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.linalg.operator import *
from dtqw.linalg.state import *

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

    def create_coin_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.logger:
            self.logger.info("building coin operator...")
        t1 = datetime.now()

        self._coin_operator = self._coin.create_operator(self._mesh, storage_level)

        app_id = self._spark_context.applicationId
        rdd_id = self._coin_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('coinOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_sparsity('coinOperator', self._coin_operator)
            self.profiler.profile_rdd('coinOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info("coin operator was built in {}s".format(self.profiler.get_times(name='coinOperator')))
                self.logger.info(
                    "coin operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='coinOperator', key='memoryUsed'),
                        self.profiler.get_rdd(name='coinOperator', key='diskUsed')
                    )
                )
                self.logger.debug("shape of coin operator: {}".format(self._coin_operator.shape))
                self.logger.debug(
                    "number of elements of coin operator: {}, which {} are nonzero".format(
                        self._coin_operator.num_elements, self._coin_operator.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of coin operator: {}".format(self._coin_operator.sparsity))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

    def create_shift_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.logger:
            self.logger.info("building shift operator...")
        t1 = datetime.now()

        self._shift_operator = self._mesh.create_operator(storage_level)

        app_id = self._spark_context.applicationId
        rdd_id = self._shift_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('shiftOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_sparsity('shiftOperator', self._shift_operator)
            self.profiler.profile_rdd('shiftOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "shift operator was built in {}s".format(self.profiler.get_times(name='shiftOperator'))
                )
                self.logger.info(
                    "shift operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='shiftOperator', key='memoryUsed'),
                        self.profiler.get_rdd(name='shiftOperator', key='diskUsed')
                    )
                )
                self.logger.debug("shape of shift operator: {}".format(self._shift_operator.shape))
                self.logger.debug(
                    "number of elements of shift operator: {}, which {} are nonzero".format(
                        self._shift_operator.num_elements, self._shift_operator.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of shift operator: {}".format(self._shift_operator.sparsity))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

    def create_unitary_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self.logger:
            self.logger.info("building unitary operator...")

        if self._coin_operator is None:
            if self.logger:
                self.logger.info("no coin operator has been set. A new one will be built")
            self.create_coin_operator(storage_level)

        if self._shift_operator is None:
            if self.logger:
                self.logger.info("no shift operator has been set. A new one will be built")
            self.create_shift_operator(storage_level)

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

        self._unitary_operator = so.multiply(co).materialize(storage_level)

        self._coin_operator.unpersist()
        self._shift_operator.unpersist()

        app_id = self._spark_context.applicationId
        rdd_id = self._unitary_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('unitaryOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_sparsity('unitaryOperator', self._unitary_operator)
            self.profiler.profile_rdd('unitaryOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "unitary operator was built in {}s".format(self.profiler.get_times(name='unitaryOperator'))
                )
                self.logger.info(
                    "unitary operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='unitaryOperator', key='memoryUsed'),
                        self.profiler.get_rdd(name='unitaryOperator', key='diskUsed')
                    )
                )
                self.logger.debug("shape of unitary operator: {}".format(self._unitary_operator.shape))
                self.logger.debug(
                    "number of elements of unitary operator: {}, which {} are nonzero".format(
                        self._unitary_operator.num_elements, self._unitary_operator.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of unitary operator: {}".format(self._unitary_operator.sparsity))

            self.profiler.log_executors(app_id=app_id)
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
        ).map(
            lambda m: (m[1], (m[0], m[2]))
        ).partitionBy(
            numPartitions=self._num_partitions
        )

        self._interaction_operator = Operator(
            self._spark_context, rdd, shape
        ).persist(storage_level).checkpoint().materialize(storage_level)

        app_id = self._spark_context.applicationId
        rdd_id = self._interaction_operator.data.id()

        if self.profiler:
            self.profiler.profile_times('interactionOperator', (datetime.now() - t1).total_seconds())
            self.profiler.profile_sparsity('interactionOperator', self._interaction_operator)
            self.profiler.profile_rdd('interactionOperator', app_id, rdd_id)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "interaction operator was built in {}s".format(self.profiler.get_times(name='interactionOperator'))
                )
                self.logger.info(
                    "interaction operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='interactionOperator', key='memoryUsed'),
                        self.profiler.get_rdd(name='interactionOperator', key='diskUsed')
                    )
                )
                self.logger.debug("shape of interaction operator: {}".format(self._interaction_operator.shape))
                self.logger.debug(
                    "number of elements of interaction operator: {}, which {} are nonzero".format(
                        self._interaction_operator.num_elements, self._interaction_operator.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of interaction operator: {}".format(self._interaction_operator.sparsity))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

    def create_walk_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        if self._unitary_operator is None:
            if self.logger:
                self.logger.info("no unitary operator has been set. A new one will be built")
            self.create_unitary_operator()

        t1 = datetime.now()

        app_id = self._spark_context.applicationId

        if self._num_particles == 1:
            if self.logger:
                self.logger.info("with just one particle, the walk operator is the unitary operator")

            rdd = self._unitary_operator.data.map(
                lambda m: (m[1], (m[0], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )

            self._walk_operator = Operator(
                self._spark_context, rdd, self._unitary_operator.shape
            ).persist(storage_level).checkpoint().materialize(storage_level)

            self._unitary_operator.unpersist()

            rdd_id = self._walk_operator.data.id()

            if self.profiler:
                self.profiler.profile_times('walkOperator', (datetime.now() - t1).total_seconds())
                self.profiler.profile_sparsity('walkOperator', self._walk_operator)
                self.profiler.profile_rdd('walkOperator', app_id, rdd_id)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "walk operator was built in {}s".format(self.profiler.get_times(name='walkOperator'))
                    )
                    self.logger.info(
                        "walk operator is consuming {} bytes in memory and {} bytes in disk".format(
                            self.profiler.get_rdd(name='walkOperator', key='memoryUsed'),
                            self.profiler.get_rdd(name='walkOperator', key='diskUsed')
                        )
                    )
                    self.logger.debug("shape of walk operator: {}".format(self._walk_operator.shape))
                    self.logger.debug(
                        "number of elements of walk operator: {}, which {} are nonzero".format(
                            self._walk_operator.num_elements, self._walk_operator.num_nonzero_elements
                        )
                    )
                    self.logger.debug("sparsity of walk operator: {}".format(self._walk_operator.sparsity))

                self.profiler.log_executors(app_id=app_id)
                self.profiler.log_rdd(app_id=app_id)
        else:
            if self.logger:
                self.logger.info("building walk operator...")

            shape = self._unitary_operator.shape
            shape_tmp = shape

            t_tmp = datetime.now()

            uo = broadcast(self._spark_context, self._unitary_operator.data.collect())

            self._walk_operator = []

            for p in range(self._num_particles):
                if self.logger:
                    self.logger.debug("building walk operator for particle {}...".format(p + 1))

                shape = shape_tmp

                if p == 0:
                    rdd = self._unitary_operator.data
                    shape_broad = broadcast(self._spark_context, (shape[0], shape[1]))

                    for p2 in range(self._num_particles - 1 - p):
                        def __map(m):
                            for i in range(shape_tmp[0]):
                                yield (m[0] * shape_broad.value[0] + i, m[1] * shape_broad.value[1] + i, m[2])

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])
                else:
                    t_tmp = datetime.now()

                    for p2 in range(p - 1):
                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                    shape_broad = broadcast(self._spark_context, (shape[0], shape[1]))

                    def __map(m):
                        for i in uo.value:
                            yield (m * shape_broad.value[0] + i[0], m * shape_broad.value[1] + i[1], i[2])

                    rdd = self._spark_context.range(
                        shape[0]
                    ).flatMap(
                        __map
                    )

                    shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                    shape_broad2 = broadcast(self._spark_context, (shape[0], shape[1]))

                    for p2 in range(self._num_particles - 1 - p):
                        def __map(m):
                            for i in range(shape_tmp[0]):
                                yield (m[0] * shape_broad2.value[0] + i, m[1] * shape_broad2.value[1] + i, m[2])

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                rdd = rdd.map(
                    lambda m: (m[1], (m[0], m[2]))
                ).partitionBy(
                    numPartitions=self._num_partitions
                )

                self._walk_operator.append(
                    Operator(
                        self._spark_context, rdd, shape
                    ).persist(storage_level).checkpoint().materialize(storage_level)
                )

                shape_broad.unpersist()
                if p > 0:
                    shape_broad2.unpersist()

                if self.profiler:
                    self.profiler.profile_times(
                        'walkOperatorParticle{}'.format(p + 1), (datetime.now() - t_tmp).total_seconds()
                    )
                    self.profiler.profile_sparsity('walkOperatorParticle{}'.format(p + 1), self._walk_operator[p])
                    self.profiler.profile_rdd(
                        'walkOperatorParticle{}'.format(p + 1), app_id, self._walk_operator[p].data.id()
                    )

                    if self.logger:
                        self.logger.info(
                            "walk operator for particle {} was built in {}s".format(
                                p + 1, self.profiler.get_times(name='walkOperatorParticle{}'.format(p + 1))
                            )
                        )
                        self.logger.info(
                            "walk operator for particle {} is consuming {} bytes in memory and {} bytes in disk".format(
                                p + 1,
                                self.profiler.get_rdd(name='walkOperatorParticle{}'.format(p + 1), key='memoryUsed'),
                                self.profiler.get_rdd(name='walkOperatorParticle{}'.format(p + 1), key='diskUsed')
                            )
                        )
                        self.logger.debug(
                            "shape of walk operator for particle {}: {}".format(p + 1, self._walk_operator[p].shape)
                        )
                        self.logger.debug(
                            "number of elements of walk operator for particle {}: {}, which {} are nonzero".format(
                                p + 1, self._walk_operator[p].num_elements, self._walk_operator[p].num_nonzero_elements
                            )
                        )
                        self.logger.debug("sparsity of walk operator for particle {}: {}".format(
                            p + 1, self._walk_operator[p].sparsity)
                        )

            uo.unpersist()
            self._unitary_operator.unpersist()

            if self.profiler:
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                self.profiler.log_executors(app_id=app_id)
                self.profiler.log_rdd(app_id=app_id)

    def title(self):
        return "Quantum Walk with {} Particle(s) on a {}".format(self._num_particles, self._mesh.title())

    def destroy_coin_operator(self):
        if self._coin_operator is not None:
            self._coin_operator.destroy()
            self._coin_operator = None

    def destroy_shift_operator(self):
        if self._shift_operator is not None:
            self._shift_operator.destroy()
            self._shift_operator = None

    def destroy_unitary_operator(self):
        if self._unitary_operator is not None:
            self._unitary_operator.destroy()
            self._unitary_operator = None

    def destroy_interaction_operator(self):
        if self._interaction_operator is not None:
            self._interaction_operator.destroy()
            self._interaction_operator = None

    def destroy_walk_operator(self):
        if self._walk_operator is not None:
            if self._num_particles == 1:
                    self._walk_operator.destroy()
            else:
                for wo in self._walk_operator:
                    wo.destroy()
            self._walk_operator = None

    def destroy_operators(self):
        if self.logger:
            self.logger.info('destroying operators...')

        self.destroy_coin_operator()
        self.destroy_shift_operator()
        self.destroy_unitary_operator()
        self.destroy_interaction_operator()
        self.destroy_walk_operator()

    def _monoparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        app_id = self._spark_context.applicationId

        # if self.logger:
        #     self.logger.debug("walk operator lineage:\n{}".format(self._walk_operator.data.toDebugString().decode()))

        result = initial_state

        if self.logger:
            self.logger.info("starting the walk...")

        for i in range(1, steps + 1, 1):
            if self._mesh.broken_links_probability:
                self.destroy_shift_operator()
                self.destroy_unitary_operator()
                self.destroy_walk_operator()
                self.create_walk_operator(storage_level)

            t_tmp = datetime.now()

            result_tmp = self._walk_operator.multiply(result).materialize(storage_level)

            result.unpersist()

            result = result_tmp

            if self.logger:
                self.logger.debug("step {} was done in {}s".format(i, (datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self.profiler:
                self.profiler.profile_rdd('systemStateStep{}'.format(i), app_id, rdd_id)
                self.profiler.profile_sparsity('systemStateStep{}'.format(i), result)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "system state after {} step(s) is consuming {} bytes in memory and {} bytes in disk".format(
                            i,
                            self.profiler.get_rdd(name='systemStateStep{}'.format(i), key='memoryUsed'),
                            self.profiler.get_rdd(name='systemStateStep{}'.format(i), key='diskUsed')
                        )
                    )
                    self.logger.debug(
                        "number of elements of system state after {} step(s): {}, which {} are nonzero".format(
                            i, result.num_elements, result.num_nonzero_elements
                        )
                    )
                    self.logger.debug("sparsity of system state after {} step(s): {}".format(i, result.sparsity))

        return result

    def _multiparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        app_id = self._spark_context.applicationId

        # for o in range(len(self._walk_operator)):
        #     if self.logger:
        #         self.logger.debug(
        #             "walk operator lineage for particle {}:\n{}".format(
        #                 o + 1, self._walk_operator[o].data.toDebugString().decode()
        #             )
        #         )
        #
        # if self.logger:
        #     self.logger.debug("interaction operator lineage:\n{}".format(
        #             self._interaction_operator.data.toDebugString().decode()
        #         )
        #     )

        result = initial_state

        if self.logger:
            self.logger.info("starting the walk...")

        for i in range(1, steps + 1, 1):
            if self._mesh.broken_links_probability:
                self.destroy_shift_operator()
                self.destroy_unitary_operator()
                self.destroy_interaction_operator()
                self.destroy_walk_operator()
                self.create_walk_operator(storage_level)

            t_tmp = datetime.now()

            result_tmp = result

            if self._interaction_operator is not None:
                result_tmp = self._interaction_operator.multiply(result_tmp)

            for wo in self._walk_operator:
                result_tmp = wo.multiply(result_tmp)

            result_tmp.materialize(storage_level)
            result.unpersist()

            result = result_tmp

            if self.logger:
                self.logger.debug("step {} was done in {}s".format(i, (datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self.profiler:
                self.profiler.profile_rdd('systemStateStep{}'.format(i), app_id, rdd_id)
                self.profiler.profile_sparsity('systemStateStep{}'.format(i), result)
                self.profiler.profile_resources(app_id)
                self.profiler.profile_executors(app_id)

                if self.logger:
                    self.logger.info(
                        "system state after {} step(s) is consuming {} bytes in memory and {} bytes in disk".format(
                            i,
                            self.profiler.get_rdd(name='systemStateStep{}'.format(i), key='memoryUsed'),
                            self.profiler.get_rdd(name='systemStateStep{}'.format(i), key='diskUsed')
                        )
                    )
                    self.logger.debug(
                        "number of elements of system state after {} step(s): {}, which {} are nonzero".format(
                            i, result.num_elements, result.num_nonzero_elements
                        )
                    )
                    self.logger.debug("sparsity of system state after {} step(s): {}".format(i, result.sparsity))

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
                if phase is None:
                    self.logger.info("no collision phase has been defined")
                elif phase == 0.0:
                    self.logger.info("a zeroed collision phase was defined. No interaction operator will be built")
                else:
                    self.logger.info("collision phase: {}".format(phase))

            if self._mesh.broken_links_probability is None:
                self.logger.info("no broken links probability has been defined")
            elif self._mesh.broken_links_probability == 0.0:
                self.logger.info("a zeroed broken links probability was defined. No decoherence will be simulated")
            else:
                self.logger.info("broken links probability: {}".format(self._mesh.broken_links_probability))

        rdd = initial_state.data.partitionBy(
            numPartitions=self._num_partitions
        )

        result = State(
            self._spark_context, rdd, initial_state.shape, self._mesh, self._num_particles
        ).materialize(storage_level)

        initial_state.unpersist()

        if not result.is_unitary():
            if self.logger:
                self.logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        app_id = self._spark_context.applicationId
        rdd_id = result.data.id()

        if self.profiler:
            self.profiler.profile_rdd('initialState', app_id, rdd_id)
            self.profiler.profile_sparsity('initialState', result)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='initialState', key='memoryUsed'),
                        self.profiler.get_rdd(name='initialState', key='diskUsed')
                    )
                )
                self.logger.debug("shape of initial state: {}".format(result.shape))
                self.logger.debug(
                    "number of elements of initial state: {}, which {} are nonzero".format(
                        result.num_elements, result.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of initial state: {}".format(result.sparsity))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

        if steps > 0:
            if not self._mesh.broken_links_probability:
                if self._walk_operator is None:
                    if self.logger:
                        self.logger.info("no walk operator has been set. A new one will be built")
                    self.create_walk_operator(storage_level)

                if self._num_particles > 1 and phase and self._interaction_operator is None:
                    if self.logger:
                        self.logger.info("no interaction operator has been set. A new one will be built")
                    self.create_interaction_operator(phase, storage_level)

            t1 = datetime.now()

            if self._num_particles == 1:
                result = self._monoparticle_walk(steps, result, storage_level)
            else:
                result = self._multiparticle_walk(steps, result, storage_level)

            if self.profiler:
                self.profiler.profile_times('walk', (datetime.now() - t1).total_seconds())

                if self.logger:
                    self.logger.info("walk was done in {}s".format(self.profiler.get_times(name='walk')))

            t1 = datetime.now()

            if self.logger:
                self.logger.debug("checking if the final state is unitary...")

            if not result.is_unitary():
                if self.logger:
                    self.logger.error("the final state is not unitary")
                raise ValueError("the final state is not unitary")

            if self.logger:
                self.logger.debug("unitarity check was done in {}s".format((datetime.now() - t1).total_seconds()))

        rdd_id = result.data.id()

        if self.profiler:
            self.profiler.profile_rdd('finalState', app_id, rdd_id)
            self.profiler.profile_sparsity('finalState', result)
            self.profiler.profile_resources(app_id)
            self.profiler.profile_executors(app_id)

            if self.logger:
                self.logger.info(
                    "final state is consuming {} bytes in memory and {} bytes in disk".format(
                        self.profiler.get_rdd(name='finalState', key='memoryUsed'),
                        self.profiler.get_rdd(name='finalState', key='diskUsed')
                    )
                )
                self.logger.debug("shape of final state: {}".format(result.shape))
                self.logger.debug(
                    "number of elements of final state: {}, which {} are nonzero".format(
                        result.num_elements, result.num_nonzero_elements
                    )
                )
                self.logger.debug("sparsity of final state: {}".format(result.sparsity))

            self.profiler.log_executors(app_id=app_id)
            self.profiler.log_rdd(app_id=app_id)

        return result
