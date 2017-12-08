import cmath
from datetime import datetime
from pyspark import StorageLevel
from dtqw.utils.utils import broadcast
from dtqw.utils.logger import is_logger
from dtqw.utils.profiler import is_profiler
from dtqw.linalg.matrix import Matrix
from dtqw.linalg.operator import *
from dtqw.linalg.state import *

__all__ = ['DiscreteTimeQuantumWalk']


class DiscreteTimeQuantumWalk:
    """Build the necessary operators and perform a discrete time quantum walk."""

    def __init__(self, spark_context, coin, mesh, num_particles, num_partitions):
        """
        Build a discrete time quantum walk object

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        coin : Coin
            A Coin object.
        mesh : Mesh
            A Mesh object.
        num_particles : int
            The number of particles present in the walk.
        num_partitions : int
            The desired number of partitions for the RDD.

        """
        self._spark_context = spark_context
        self._coin = coin
        self._mesh = mesh
        self._num_particles = num_particles
        self._num_partitions = num_partitions

        self._coin_operator = None
        self._shift_operator = None
        self._evolution_operator = None
        self._interaction_operator = None
        self._walk_operator = None

        if num_particles < 1:
            # self._logger.error("Invalid number of particles")
            raise ValueError("invalid number of particles")

        self._logger = None
        self._profiler = None

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
    def evolution_operator(self):
        return self._evolution_operator

    @property
    def interaction_operator(self):
        return self._interaction_operator

    @property
    def walk_operator(self):
        return self._walk_operator

    @property
    def logger(self):
        return self._logger

    @property
    def profiler(self):
        return self._profiler

    @coin_operator.setter
    def coin_operator(self, co):
        """
        Parameters
        ----------
        co : Operator

        Raises
        ------
        TypeError

        """
        if is_operator(co):
            self._coin_operator = co
        else:
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(co)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(co)))

    @shift_operator.setter
    def shift_operator(self, so):
        """
        Parameters
        ----------
        so : Operator

        Raises
        ------
        TypeError

        """
        if is_operator(so):
            self._shift_operator = so
        else:
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(so)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(so)))

    @evolution_operator.setter
    def evolution_operator(self, uo):
        """
        Parameters
        ----------
        uo : Operator

        Raises
        ------
        TypeError

        """
        if is_operator(uo):
            self._evolution_operator = uo
        else:
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(uo)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(uo)))

    @interaction_operator.setter
    def interaction_operator(self, io):
        """
        Parameters
        ----------
        io : Operator

        Raises
        ------
        TypeError

        """
        if is_operator(io) or io is None:
            self._interaction_operator = io
        else:
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(io)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(io)))

    @walk_operator.setter
    def walk_operator(self, wo):
        """
        Parameters
        ----------
        wo : Operator or list of Operator
            An Operator or a list of Operators (for multiparticle walk simulator).

        Raises
        ------
        ValueError
        TypeError

        """
        if is_operator(wo) or wo is None:
            self._walk_operator = wo
        elif isinstance(wo, (list, tuple)):
            if len(wo) != self._num_particles:
                if self._logger:
                    self._logger.error('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))
                raise ValueError('{} walk operators expected, not {}'.format(self._num_particles, len(wo)))

            for o in wo:
                if not is_operator(o):
                    if self._logger:
                        self._logger.error('Operator instance expected, not "{}"'.format(type(wo)))
                    raise TypeError('Operator instance expected, not "{}"'.format(type(wo)))
        else:
            if self._logger:
                self._logger.error('Operator instance expected, not "{}"'.format(type(wo)))
            raise TypeError('Operator instance expected, not "{}"'.format(type(wo)))

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

    def to_string(self):
        return self.__str__()

    def title(self):
        return "Quantum Walk with {} Particle(s) on a {}".format(self._num_particles, self._mesh.title())

    def create_evolution_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the evolution operator for the walk.

        Parameters
        ----------
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        """
        if self._logger:
            self._logger.info("building evolution operator...")

        if self._coin_operator is None:
            if self._logger:
                self._logger.info("no coin operator has been set. A new one will be built")
            self._coin_operator = self._coin.create_operator(
                self._mesh, self._num_partitions, Matrix.CoordinateMultiplicand, storage_level
            )

        if self._shift_operator is None:
            if self._logger:
                self._logger.info("no shift operator has been set. A new one will be built")
            self._shift_operator = self._mesh.create_operator(
                self._num_partitions, Matrix.CoordinateMultiplier, storage_level
            )

        t1 = datetime.now()

        self._evolution_operator = self._shift_operator.multiply(self._coin_operator).materialize(storage_level)

        self._coin_operator.unpersist()
        self._shift_operator.unpersist()

        app_id = self._spark_context.applicationId
        rdd_id = self._evolution_operator.data.id()

        if self._profiler:
            self._profiler.profile_times('evolutionOperator', (datetime.now() - t1).total_seconds())
            self._profiler.profile_sparsity('evolutionOperator', self._evolution_operator)
            self._profiler.profile_rdd('evolutionOperator', app_id, rdd_id)
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            if self._logger:
                self._logger.info(
                    "evolution operator was built in {}s".format(self._profiler.get_times(name='evolutionOperator'))
                )
                self._logger.info(
                    "evolution operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self._profiler.get_rdd(name='evolutionOperator', key='memoryUsed'),
                        self._profiler.get_rdd(name='evolutionOperator', key='diskUsed')
                    )
                )
                self._logger.debug("shape of evolution operator: {}".format(self._evolution_operator.shape))
                self._logger.debug(
                    "number of elements of evolution operator: {}, which {} are nonzero".format(
                        self._evolution_operator.num_elements, self._evolution_operator.num_nonzero_elements
                    )
                )
                self._logger.debug("sparsity of evolution operator: {}".format(self._evolution_operator.sparsity))

    def create_interaction_operator(self, phase, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the particles' interaction operator for the walk.

        Parameters
        ----------
        phase : float
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        """
        if self._logger:
            self._logger.info("building interaction operator...")

        t1 = datetime.now()

        phase = cmath.exp(phase * (0.0 + 1.0j))
        num_particles = self._num_particles

        coin_size = 2

        if self._mesh.is_1d():
            size = self._mesh.size
            cs_size = coin_size * size

            rdd_range = cs_size ** num_particles
            shape = (rdd_range, rdd_range)

            def __map(m):
                a = []

                for p in range(num_particles):
                    a.append(int(m / (cs_size ** (num_particles - 1 - p))) % size)

                for p in range(num_particles):
                    if a[0] != a[p]:
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
            if self._logger:
                self._logger.error("mesh dimension not implemented")
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

        if self._profiler:
            self._profiler.profile_times('interactionOperator', (datetime.now() - t1).total_seconds())
            self._profiler.profile_sparsity('interactionOperator', self._interaction_operator)
            self._profiler.profile_rdd('interactionOperator', app_id, rdd_id)
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            if self._logger:
                self._logger.info(
                    "interaction operator was built in {}s".format(self._profiler.get_times(name='interactionOperator'))
                )
                self._logger.info(
                    "interaction operator is consuming {} bytes in memory and {} bytes in disk".format(
                        self._profiler.get_rdd(name='interactionOperator', key='memoryUsed'),
                        self._profiler.get_rdd(name='interactionOperator', key='diskUsed')
                    )
                )
                self._logger.debug("shape of interaction operator: {}".format(self._interaction_operator.shape))
                self._logger.debug(
                    "number of elements of interaction operator: {}, which {} are nonzero".format(
                        self._interaction_operator.num_elements, self._interaction_operator.num_nonzero_elements
                    )
                )
                self._logger.debug("sparsity of interaction operator: {}".format(self._interaction_operator.sparsity))

    def create_walk_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the walk operator for the walk.

        When performing a multiparticle walk, this method builds a list with n operators,
        where n is the number of particles of the system. In this case, each operator is built by
        applying a tensor product between the evolution operator and n-1 identity matrices as follows:

            W1 = W (X) I2 (X) ... (X) In
            Wi = I1 (X) ... (X) Ii-1 (X) Wi (X) Ii+1 (X) ... In
            Wn = I1 (X) ... (X) In-1 (X) W

        Regardless the number of particles, the walk operators have their (i,j,value) coordinates converted to
        apropriate coordinates for multiplication, in this case, the Operator.MultiplierCoordinate.

        Parameters
        ----------
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        """
        if self._evolution_operator is None:
            if self._logger:
                self._logger.info("no evolution operator has been set. A new one will be built")
            self.create_evolution_operator()

        t1 = datetime.now()

        app_id = self._spark_context.applicationId

        if self._num_particles == 1:
            if self._logger:
                self._logger.info("with just one particle, the walk operator is the evolution operator")

            # Converting to an apropriate coordinate for multiplication
            rdd = self._evolution_operator.data.map(
                lambda m: (m[1], (m[0], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )

            self._walk_operator = Operator(
                self._spark_context, rdd, self._evolution_operator.shape
            ).persist(storage_level).checkpoint().materialize(storage_level)

            self._evolution_operator.unpersist()

            rdd_id = self._walk_operator.data.id()

            if self._profiler:
                self._profiler.profile_times('walkOperator', (datetime.now() - t1).total_seconds())
                self._profiler.profile_sparsity('walkOperator', self._walk_operator)
                self._profiler.profile_rdd('walkOperator', app_id, rdd_id)
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

                if self._logger:
                    self._logger.info(
                        "walk operator was built in {}s".format(self._profiler.get_times(name='walkOperator'))
                    )
                    self._logger.info(
                        "walk operator is consuming {} bytes in memory and {} bytes in disk".format(
                            self._profiler.get_rdd(name='walkOperator', key='memoryUsed'),
                            self._profiler.get_rdd(name='walkOperator', key='diskUsed')
                        )
                    )
                    self._logger.debug("shape of walk operator: {}".format(self._walk_operator.shape))
                    self._logger.debug(
                        "number of elements of walk operator: {}, which {} are nonzero".format(
                            self._walk_operator.num_elements, self._walk_operator.num_nonzero_elements
                        )
                    )
                    self._logger.debug("sparsity of walk operator: {}".format(self._walk_operator.sparsity))
        else:
            if self._logger:
                self._logger.info("building walk operator...")

            shape = self._evolution_operator.shape
            shape_tmp = shape

            t_tmp = datetime.now()

            uo = broadcast(self._spark_context, self._evolution_operator.data.collect())

            self._walk_operator = []

            for p in range(self._num_particles):
                if self._logger:
                    self._logger.debug("building walk operator for particle {}...".format(p + 1))

                shape = shape_tmp

                if p == 0:
                    rdd = self._evolution_operator.data

                    # The first particle's walk operator consists in applying the tensor product between the
                    # evolution operator and the other particles' corresponding identity matrices
                    #
                    # W1 = W (X) I2 (X) ... (X) In
                    for p2 in range(self._num_particles - 1 - p):
                        def __map(m):
                            for i in range(shape_tmp[0]):
                                yield (m[0] * shape_tmp[0] + i, m[1] * shape_tmp[1] + i, m[2])

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])
                else:
                    t_tmp = datetime.now()

                    # For the other particles, each one has its operator built by applying the
                    # tensor product between its previous particles' identity matrices and its evolution operator.
                    #
                    # Wi = I1 (X) ... (X) Ii-1 (X) Wi ...
                    for p2 in range(p - 1):
                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                    def __map(m):
                        for i in uo.value:
                            yield (m * shape_tmp[0] + i[0], m * shape_tmp[1] + i[1], i[2])

                    rdd = self._spark_context.range(
                        shape[0]
                    ).flatMap(
                        __map
                    )

                    shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                    # Then, the tensor product is applied between the following particles' identity matrices.
                    #
                    # ... (X) Ii+1 (X) ... In
                    for p2 in range(self._num_particles - 1 - p):
                        def __map(m):
                            for i in range(shape_tmp[0]):
                                yield (m[0] * shape_tmp[0] + i, m[1] * shape_tmp[1] + i, m[2])

                        rdd = rdd.flatMap(
                            __map
                        )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                # Converting to an apropriate coordinate for multiplication
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

                if self._profiler:
                    self._profiler.profile_times(
                        'walkOperatorParticle{}'.format(p + 1), (datetime.now() - t_tmp).total_seconds()
                    )
                    self._profiler.profile_sparsity('walkOperatorParticle{}'.format(p + 1), self._walk_operator[p])
                    self._profiler.profile_rdd(
                        'walkOperatorParticle{}'.format(p + 1), app_id, self._walk_operator[p].data.id()
                    )

                    if self._logger:
                        self._logger.info(
                            "walk operator for particle {} was built in {}s".format(
                                p + 1, self._profiler.get_times(name='walkOperatorParticle{}'.format(p + 1))
                            )
                        )
                        self._logger.info(
                            "walk operator for particle {} is consuming {} bytes in memory and {} bytes in disk".format(
                                p + 1,
                                self._profiler.get_rdd(name='walkOperatorParticle{}'.format(p + 1), key='memoryUsed'),
                                self._profiler.get_rdd(name='walkOperatorParticle{}'.format(p + 1), key='diskUsed')
                            )
                        )
                        self._logger.debug(
                            "shape of walk operator for particle {}: {}".format(p + 1, self._walk_operator[p].shape)
                        )
                        self._logger.debug(
                            "number of elements of walk operator for particle {}: {}, which {} are nonzero".format(
                                p + 1, self._walk_operator[p].num_elements, self._walk_operator[p].num_nonzero_elements
                            )
                        )
                        self._logger.debug("sparsity of walk operator for particle {}: {}".format(
                            p + 1, self._walk_operator[p].sparsity)
                        )

            uo.unpersist()
            self._evolution_operator.unpersist()

            if self._profiler:
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

    def destroy_coin_operator(self):
        """Call the Operator's method destroy."""
        if self._coin_operator is not None:
            self._coin_operator.destroy()
            self._coin_operator = None

    def destroy_shift_operator(self):
        """Call the Operator's method destroy."""
        if self._shift_operator is not None:
            self._shift_operator.destroy()
            self._shift_operator = None

    def destroy_evolution_operator(self):
        """Call the Operator's method destroy."""
        if self._evolution_operator is not None:
            self._evolution_operator.destroy()
            self._evolution_operator = None

    def destroy_interaction_operator(self):
        """Call the Operator's method destroy."""
        if self._interaction_operator is not None:
            self._interaction_operator.destroy()
            self._interaction_operator = None

    def destroy_walk_operator(self):
        """Call the Operator's method destroy."""
        if self._walk_operator is not None:
            if self._num_particles == 1:
                    self._walk_operator.destroy()
            else:
                for wo in self._walk_operator:
                    wo.destroy()
            self._walk_operator = None

    def destroy_operators(self):
        """Release all operators from memory."""
        if self._logger:
            self._logger.info('destroying operators...')

        self.destroy_coin_operator()
        self.destroy_shift_operator()
        self.destroy_evolution_operator()
        self.destroy_interaction_operator()
        self.destroy_walk_operator()

    def _monoparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        app_id = self._spark_context.applicationId

        result = initial_state

        if self._logger:
            self._logger.info("starting the walk...")

        for i in range(1, steps + 1, 1):
            if self._mesh.broken_links_probability:
                self.destroy_shift_operator()
                self.destroy_evolution_operator()
                self.destroy_walk_operator()
                self.create_walk_operator(storage_level)

            t_tmp = datetime.now()

            result_tmp = self._walk_operator.multiply(result).materialize(storage_level)

            result.unpersist()

            result = result_tmp

            if self._logger:
                self._logger.debug("step {} was done in {}s".format(i, (datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self._profiler:
                self._profiler.profile_rdd('systemStateStep{}'.format(i), app_id, rdd_id)
                self._profiler.profile_sparsity('systemStateStep{}'.format(i), result)
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

                if self._logger:
                    self._logger.info(
                        "system state after {} step(s) is consuming {} bytes in memory and {} bytes in disk".format(
                            i,
                            self._profiler.get_rdd(name='systemStateStep{}'.format(i), key='memoryUsed'),
                            self._profiler.get_rdd(name='systemStateStep{}'.format(i), key='diskUsed')
                        )
                    )
                    self._logger.debug(
                        "number of elements of system state after {} step(s): {}, which {} are nonzero".format(
                            i, result.num_elements, result.num_nonzero_elements
                        )
                    )
                    self._logger.debug("sparsity of system state after {} step(s): {}".format(i, result.sparsity))

        return result

    def _multiparticle_walk(self, steps, initial_state, storage_level=StorageLevel.MEMORY_AND_DISK):
        app_id = self._spark_context.applicationId

        result = initial_state

        if self._logger:
            self._logger.info("starting the walk...")

        for i in range(1, steps + 1, 1):
            if self._mesh.broken_links_probability:
                self.destroy_shift_operator()
                self.destroy_evolution_operator()
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

            if self._logger:
                self._logger.debug("step {} was done in {}s".format(i, (datetime.now() - t_tmp).total_seconds()))

            rdd_id = result.data.id()

            if self._profiler:
                self._profiler.profile_rdd('systemStateStep{}'.format(i), app_id, rdd_id)
                self._profiler.profile_sparsity('systemStateStep{}'.format(i), result)
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

                if self._logger:
                    self._logger.info(
                        "system state after {} step(s) is consuming {} bytes in memory and {} bytes in disk".format(
                            i,
                            self._profiler.get_rdd(name='systemStateStep{}'.format(i), key='memoryUsed'),
                            self._profiler.get_rdd(name='systemStateStep{}'.format(i), key='diskUsed')
                        )
                    )
                    self._logger.debug(
                        "number of elements of system state after {} step(s): {}, which {} are nonzero".format(
                            i, result.num_elements, result.num_nonzero_elements
                        )
                    )
                    self._logger.debug("sparsity of system state after {} step(s): {}".format(i, result.sparsity))

        return result

    def walk(self, steps, initial_state, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform a walk

        Parameters
        ----------
        steps : int
        initial_state : State
            The initial state of the system.
        phase : float
        storage_level : StorageLevel
            The desired storage level when materializing the RDD.

        Returns
        -------
        State
            The final state of the system after performing the walk.

        """
        if not self._mesh.check_steps(steps):
            if self._logger:
                self._logger.error("invalid number of steps")
            raise ValueError("invalid number of steps")

        if self._logger:
            self._logger.info("steps: {}".format(steps))
            self._logger.info("space size: {}".format(self._mesh.size))
            self._logger.info("number of particles: {}".format(self._num_particles))
            self._logger.info("number of partitions: {}".format(self._num_partitions))

            if self._num_particles > 1:
                if phase is None:
                    self._logger.info("no collision phase has been defined")
                elif phase == 0.0:
                    self._logger.info("a zeroed collision phase was defined. No interaction operator will be built")
                else:
                    self._logger.info("collision phase: {}".format(phase))

            if self._mesh.broken_links_probability is None:
                self._logger.info("no broken links probability has been defined")
            elif self._mesh.broken_links_probability == 0.0:
                self._logger.info("a zeroed broken links probability was defined. No decoherence will be simulated")
            else:
                self._logger.info("broken links probability: {}".format(self._mesh.broken_links_probability))

        # Partitioning the initial state of the system in order to reduce some shuffling operations
        rdd = initial_state.data.partitionBy(
            numPartitions=self._num_partitions
        )

        result = State(
            self._spark_context, rdd, initial_state.shape, self._mesh, self._num_particles
        ).materialize(storage_level)

        initial_state.unpersist()

        if not result.is_unitary():
            if self._logger:
                self._logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        app_id = self._spark_context.applicationId
        rdd_id = result.data.id()

        if self._profiler:
            self._profiler.profile_rdd('initialState', app_id, rdd_id)
            self._profiler.profile_sparsity('initialState', result)
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            if self._logger:
                self._logger.info(
                    "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                        self._profiler.get_rdd(name='initialState', key='memoryUsed'),
                        self._profiler.get_rdd(name='initialState', key='diskUsed')
                    )
                )
                self._logger.debug("shape of initial state: {}".format(result.shape))
                self._logger.debug(
                    "number of elements of initial state: {}, which {} are nonzero".format(
                        result.num_elements, result.num_nonzero_elements
                    )
                )
                self._logger.debug("sparsity of initial state: {}".format(result.sparsity))

        if steps > 0:
            # Building operators once if not simulating decoherence with broken links
            # When there is a broken links probability, the operators will be built in each step of the walk
            if not self._mesh.broken_links_probability:
                if self._walk_operator is None:
                    if self._logger:
                        self._logger.info("no walk operator has been set. A new one will be built")
                    self.create_walk_operator(storage_level)

                if self._num_particles > 1 and phase and self._interaction_operator is None:
                    if self._logger:
                        self._logger.info("no interaction operator has been set. A new one will be built")
                    self.create_interaction_operator(phase, storage_level)

            t1 = datetime.now()

            if self._num_particles == 1:
                result = self._monoparticle_walk(steps, result, storage_level)
            else:
                result = self._multiparticle_walk(steps, result, storage_level)

            if self._profiler:
                self._profiler.profile_times('walk', (datetime.now() - t1).total_seconds())

                if self._logger:
                    self._logger.info("walk was done in {}s".format(self._profiler.get_times(name='walk')))

            t1 = datetime.now()

            if self._logger:
                self._logger.debug("checking if the final state is evolution...")

            if not result.is_unitary():
                if self._logger:
                    self._logger.error("the final state is not evolution")
                raise ValueError("the final state is not evolution")

            if self._logger:
                self._logger.debug("unitarity check was done in {}s".format((datetime.now() - t1).total_seconds()))

        rdd_id = result.data.id()

        if self._profiler:
            self._profiler.profile_rdd('finalState', app_id, rdd_id)
            self._profiler.profile_sparsity('finalState', result)
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            if self._logger:
                self._logger.info(
                    "final state is consuming {} bytes in memory and {} bytes in disk".format(
                        self._profiler.get_rdd(name='finalState', key='memoryUsed'),
                        self._profiler.get_rdd(name='finalState', key='diskUsed')
                    )
                )
                self._logger.debug("shape of final state: {}".format(result.shape))
                self._logger.debug(
                    "number of elements of final state: {}, which {} are nonzero".format(
                        result.num_elements, result.num_nonzero_elements
                    )
                )
                self._logger.debug("sparsity of final state: {}".format(result.sparsity))

        return result
