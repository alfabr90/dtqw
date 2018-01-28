import cmath
import fileinput
from glob import glob
from datetime import datetime

from pyspark import StorageLevel

from dtqw.math.operator import Operator, is_operator
from dtqw.math.state import State
from dtqw.utils.logger import is_logger
from dtqw.utils.profiling.profiler import is_profiler
from dtqw.utils.utils import broadcast, get_tmp_path, remove_path, \
    CoordinateDefault, CoordinateMultiplier, CoordinateMultiplicand

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

    def create_interaction_operator(self, phase,
                                    coord_format=CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the particles' interaction operator for the walk.

        Parameters
        ----------
        phase : float
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is utils.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD.

        """
        if self._logger:
            self._logger.info("building interaction operator...")

        t1 = datetime.now()

        phase = cmath.exp(phase * (0.0+1.0j))
        num_particles = self._num_particles

        coin_size = 2

        if self._mesh.is_1d():
            size = self._mesh.size
            cs_size = coin_size * size

            rdd_range = cs_size ** num_particles
            shape = (rdd_range, rdd_range)

            def __map(m):
                x = []

                for p in range(num_particles):
                    x.append(int(m / (cs_size ** (num_particles - 1 - p))) % size)

                for p in range(num_particles):
                    if x[0] != x[p]:
                        return m, m, 1

                return m, m, phase
        elif self._mesh.is_2d():
            size_x = self._mesh.size[0]
            size_y = self._mesh.size[1]
            cs_size_x = coin_size * size_x
            cs_size_y = coin_size * size_y
            cs_size_xy = cs_size_x * cs_size_y

            rdd_range = cs_size_xy ** num_particles
            shape = (rdd_range, rdd_range)

            def __map(m):
                xy = []

                for p in range(num_particles):
                    xy.append(
                        (
                            int(m / (cs_size_xy ** (num_particles - 1 - p) * size_y)) % size_x,
                            int(m / (cs_size_xy ** (num_particles - 1 - p))) % size_y
                        )
                    )

                for p in range(num_particles):
                    if xy[0][0] != xy[p][0] or xy[0][1] != xy[p][1]:
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
        )

        if coord_format == CoordinateMultiplier:
            rdd = rdd.map(
                lambda m: (m[1], (m[0], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )
        elif coord_format == CoordinateMultiplicand:
            rdd = rdd.map(
                lambda m: (m[0], (m[1], m[2]))
            ).partitionBy(
                numPartitions=self._num_partitions
            )

        self._interaction_operator = Operator(
            rdd, shape, coord_format=coord_format
        ).persist(storage_level).checkpoint().materialize(storage_level)

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                'interactionOperator', self._interaction_operator, (datetime.now() - t1).total_seconds()
            )

            if self._logger:
                self._logger.info("interaction operator was built in {}s".format(info['buildingTime']))
                self._logger.info(
                    "interaction operator is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

    def create_walk_operator(self, coord_format=CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the walk operator for the walk.

        When performing a multiparticle walk, this method builds a list with n operators,
        where n is the number of particles of the system. In this case, each operator is built by
        applying a tensor product between the evolution operator and n-1 identity matrices as follows:

            W1 = W (X) I2 (X) ... (X) In
            Wi = I1 (X) ... (X) Ii-1 (X) Wi (X) Ii+1 (X) ... In
            Wn = I1 (X) ... (X) In-1 (X) W

        Regardless the number of particles, the walk operators have their (i,j,value) coordinates converted to
        appropriate coordinates for multiplication, in this case, the Matrix.MultiplierCoordinate.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is utils.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD.

        """
        app_id = self._spark_context.applicationId

        if self._coin_operator is None:
            if self._logger:
                self._logger.info("no coin operator has been set. A new one will be built")
            self._coin_operator = self._coin.create_operator(
                self._mesh, self._num_partitions, coord_format=CoordinateMultiplicand, storage_level=storage_level
            )

        if self._shift_operator is None:
            if self._logger:
                self._logger.info("no shift operator has been set. A new one will be built")
            self._shift_operator = self._mesh.create_operator(
                self._num_partitions, coord_format=CoordinateMultiplier, storage_level=storage_level
            )

        if self._num_particles == 1:
            if self._logger:
                self._logger.info("with just one particle, the walk operator is the evolution operator")

            t1 = datetime.now()

            evolution_operator = self._shift_operator.multiply(self._coin_operator, coord_format=CoordinateMultiplier)

            self._walk_operator = evolution_operator.persist(storage_level).checkpoint().materialize(storage_level)

            evolution_operator.unpersist()
            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            if self._profiler:
                self._profiler.profile_resources(app_id)
                self._profiler.profile_executors(app_id)

                info = self._profiler.profile_operator(
                    'walkOperator', self._walk_operator, (datetime.now() - t1).total_seconds()
                )

                if self._logger:
                    self._logger.info("walk operator was built in {}s".format(info['buildingTime']))
                    self._logger.info(
                        "walk operator is consuming {} bytes in memory and {} bytes in disk".format(
                            info['memoryUsed'], info['diskUsed']
                        )
                    )
        else:
            if self._logger:
                self._logger.info("building walk operator...")

            t_tmp = datetime.now()

            evolution_operator = self._shift_operator.multiply(
                self._coin_operator, coord_format=CoordinateDefault
            ).persist(storage_level).materialize(storage_level)

            self._coin_operator.unpersist()
            self._shift_operator.unpersist()

            shape = evolution_operator.shape
            shape_tmp = shape

            self._walk_operator = []

            mode = 1

            if mode == 0:
                eo = broadcast(self._spark_context, evolution_operator.data.collect())
                evolution_operator.unpersist()

                for p in range(self._num_particles):
                    if self._logger:
                        self._logger.debug("building walk operator for particle {}...".format(p + 1))

                    shape = shape_tmp

                    if p == 0:
                        # The first particle's walk operator consists in applying the tensor product between the
                        # evolution operator and the other particles' corresponding identity matrices
                        #
                        # W1 = U (X) I2 (X) ... (X) In
                        for p2 in range(self._num_particles - 2 - p):
                            shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        def __map(m):
                            for i in eo.value:
                                yield i[0] * shape[0] + m, i[1] * shape[1] + m, i[2]

                        rdd = self._spark_context.range(
                            shape[0]
                        ).flatMap(
                            __map
                        )

                        if coord_format == CoordinateMultiplier:
                            rdd = rdd.map(
                                lambda m: (m[1], (m[0], m[2]))
                            ).partitionBy(
                                numPartitions=self._num_partitions
                            )
                        elif coord_format == CoordinateMultiplicand:
                            rdd = rdd.map(
                                lambda m: (m[0], (m[1], m[2]))
                            ).partitionBy(
                                numPartitions=self._num_partitions
                            )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        self._walk_operator.append(
                            Operator(
                                rdd, shape, coord_format=coord_format
                            ).persist(storage_level).checkpoint().materialize(storage_level)
                        )
                    else:
                        t_tmp = datetime.now()

                        # For the other particles, each one has its operator built by applying the
                        # tensor product between its previous particles' identity matrices and its evolution operator.
                        #
                        # Wi = I1 (X) ... (X) Ii-1 (X) U ...
                        for p2 in range(p - 1):
                            shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        def __map(m):
                            for i in eo.value:
                                yield m * shape_tmp[0] + i[0], m * shape_tmp[1] + i[1], i[2]

                        rdd_pre = self._spark_context.range(
                            shape[0]
                        ).flatMap(
                            __map
                        )

                        new_shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        # Then, the tensor product is applied between the following particles' identity matrices.
                        #
                        # ... (X) Ii+1 (X) ... In
                        #
                        # If it is the last particle, the tensor product is applied between
                        # the pre-identity and evolution operators
                        #
                        # ... (X) Ii-1 (X) U
                        if p == self._num_particles - 1:
                            if coord_format == CoordinateMultiplier:
                                rdd_pre = rdd_pre.map(
                                    lambda m: (m[1], (m[0], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )
                            elif coord_format == CoordinateMultiplicand:
                                rdd_pre = rdd_pre.map(
                                    lambda m: (m[0], (m[1], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )

                            self._walk_operator.append(
                                Operator(
                                    rdd_pre, new_shape, coord_format=coord_format
                                ).persist(storage_level).checkpoint().materialize(storage_level)
                            )
                        else:
                            shape = shape_tmp

                            for p2 in range(self._num_particles - 2 - p):
                                shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                            def __map(m):
                                for i in range(shape[0]):
                                    yield m[0] * shape[0] + i, m[1] * shape[1] + i, m[2]

                            rdd_pos = rdd_pre.flatMap(
                                __map
                            )

                            if coord_format == CoordinateMultiplier:
                                rdd_pos = rdd_pos.map(
                                    lambda m: (m[1], (m[0], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )
                            elif coord_format == CoordinateMultiplicand:
                                rdd_pos = rdd_pos.map(
                                    lambda m: (m[0], (m[1], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )

                            new_shape = (shape[0] * new_shape[0], shape[1] * new_shape[1])

                            self._walk_operator.append(
                                Operator(
                                    rdd_pos, new_shape, coord_format=coord_format
                                ).persist(storage_level).checkpoint().materialize(storage_level)
                            )

                    if self._profiler:
                        self._profiler.profile_resources(app_id)
                        self._profiler.profile_executors(app_id)

                        info = self._profiler.profile_operator(
                            'walkOperatorParticle{}'.format(p + 1),
                            self._walk_operator[-1], (datetime.now() - t_tmp).total_seconds()
                        )

                        if self._logger:
                            self._logger.info(
                                "walk operator for particle {} was built in {}s".format(p + 1, info['buildingTime'])
                            )
                            self._logger.info(
                                "walk operator for particle {} is consuming {} bytes in memory and {} bytes in disk".format(
                                    p + 1, info['memoryUsed'], info['diskUsed']
                                )
                            )

                eo.unpersist()
            else:
                path = get_tmp_path()
                print(path)
                evolution_operator.dump(path)

                for p in range(self._num_particles):
                    if self._logger:
                        self._logger.debug("building walk operator for particle {}...".format(p + 1))

                    shape = shape_tmp

                    if p == 0:
                        # The first particle's walk operator consists in applying the tensor product between the
                        # evolution operator and the other particles' corresponding identity matrices
                        #
                        # W1 = U (X) I2 (X) ... (X) In
                        for p2 in range(self._num_particles - 2 - p):
                            shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield int(l[0]) * shape[0] + m, int(l[1]) * shape[1] + m, complex(l[2])

                        rdd = self._spark_context.range(
                            shape[0]
                        ).flatMap(
                            __map
                        )

                        if coord_format == CoordinateMultiplier:
                            rdd = rdd.map(
                                lambda m: (m[1], (m[0], m[2]))
                            ).partitionBy(
                                numPartitions=self._num_partitions
                            )
                        elif coord_format == CoordinateMultiplicand:
                            rdd = rdd.map(
                                lambda m: (m[0], (m[1], m[2]))
                            ).partitionBy(
                                numPartitions=self._num_partitions
                            )

                        shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        self._walk_operator.append(
                            Operator(
                                rdd, shape, coord_format=coord_format
                            ).persist(storage_level).checkpoint().materialize(storage_level)
                        )
                    else:
                        t_tmp = datetime.now()

                        # For the other particles, each one has its operator built by applying the
                        # tensor product between its previous particles' identity matrices and its evolution operator.
                        #
                        # Wi = I1 (X) ... (X) Ii-1 (X) U ...
                        for p2 in range(p - 1):
                            shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        def __map(m):
                            with fileinput.input(files=glob(path + '/part-*')) as f:
                                for line in f:
                                    l = line.split()
                                    yield m * shape_tmp[0] + int(l[0]), m * shape_tmp[1] + int(l[1]), complex(l[2])

                        rdd_pre = self._spark_context.range(
                            shape[0]
                        ).flatMap(
                            __map
                        )

                        new_shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                        # Then, the tensor product is applied between the following particles' identity matrices.
                        #
                        # ... (X) Ii+1 (X) ... In
                        #
                        # If it is the last particle, the tensor product is applied between
                        # the pre-identity and evolution operators
                        #
                        # ... (X) Ii-1 (X) U
                        if p == self._num_particles - 1:
                            if coord_format == CoordinateMultiplier:
                                rdd_pre = rdd_pre.map(
                                    lambda m: (m[1], (m[0], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )
                            elif coord_format == CoordinateMultiplicand:
                                rdd_pre = rdd_pre.map(
                                    lambda m: (m[0], (m[1], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )

                            self._walk_operator.append(
                                Operator(
                                    rdd_pre, new_shape, coord_format=coord_format
                                ).persist(storage_level).checkpoint().materialize(storage_level)
                            )
                        else:
                            shape = shape_tmp

                            for p2 in range(self._num_particles - 2 - p):
                                shape = (shape[0] * shape_tmp[0], shape[1] * shape_tmp[1])

                            def __map(m):
                                for i in range(shape[0]):
                                    yield m[0] * shape[0] + i, m[1] * shape[1] + i, m[2]

                            rdd_pos = rdd_pre.flatMap(
                                __map
                            )

                            if coord_format == CoordinateMultiplier:
                                rdd_pos = rdd_pos.map(
                                    lambda m: (m[1], (m[0], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )
                            elif coord_format == CoordinateMultiplicand:
                                rdd_pos = rdd_pos.map(
                                    lambda m: (m[0], (m[1], m[2]))
                                ).partitionBy(
                                    numPartitions=self._num_partitions
                                )

                            new_shape = (shape[0] * new_shape[0], shape[1] * new_shape[1])

                            self._walk_operator.append(
                                Operator(
                                    rdd_pos, new_shape, coord_format=coord_format
                                ).persist(storage_level).checkpoint().materialize(storage_level)
                            )

                    if self._profiler:
                        self._profiler.profile_resources(app_id)
                        self._profiler.profile_executors(app_id)

                        info = self._profiler.profile_operator(
                            'walkOperatorParticle{}'.format(p + 1),
                            self._walk_operator[-1], (datetime.now() - t_tmp).total_seconds()
                        )

                        if self._logger:
                            self._logger.info(
                                "walk operator for particle {} was built in {}s".format(p + 1, info['buildingTime'])
                            )
                            self._logger.info(
                                "walk operator for particle {} is consuming {} bytes in memory and {} bytes in disk".format(
                                    p + 1, info['memoryUsed'], info['diskUsed']
                                )
                            )

                evolution_operator.unpersist()
                remove_path(path)

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
        """Release all operators from memory and/or disk."""
        if self._logger:
            self._logger.info('destroying operators...')

        self.destroy_coin_operator()
        self.destroy_shift_operator()
        self.destroy_interaction_operator()
        self.destroy_walk_operator()

    def walk(self, steps, initial_state, phase=None, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Perform a walk

        Parameters
        ----------
        steps : int
        initial_state : State
            The initial state of the system.
        phase : float, optional
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD.

        Returns
        -------
        State
            The final state of the system after performing the walk.

        Raises
        ------
        ValueError

        """
        if not self._mesh.check_steps(steps):
            if self._logger:
                self._logger.error("invalid number of steps for the chosen mesh")
            raise ValueError("invalid number of steps for the chosen mesh")

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

            if self._mesh.broken_links is None:
                self._logger.info("no broken links has been defined")
            else:
                self._logger.info("broken links probability: {}".format(self._mesh.broken_links.probability))

        # Partitioning the initial state of the system in order to reduce some shuffling operations
        '''rdd = initial_state.data.partitionBy(
            numPartitions=self._num_partitions
        )

        result = State(
            self._spark_context, rdd, initial_state.shape, self._mesh, self._num_particles
        ).materialize(storage_level)

        initial_state.unpersist()'''

        result = initial_state.materialize(storage_level)

        if not result.is_unitary():
            if self._logger:
                self._logger.error("the initial state is not unitary")
            raise ValueError("the initial state is not unitary")

        app_id = self._spark_context.applicationId

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_state('initialState', result, 0.0)

            if self._logger:
                self._logger.info(
                    "initial state is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

        if self._logger:
            self._profiler.log_rdd(app_id=app_id)

        if steps > 0:
            # Building walk operators once if not simulating decoherence with broken links
            # When there is a broken links probability, the walk operators will be built in each step of the walk
            if not self._mesh.broken_links:
                if self._walk_operator is None:
                    if self._logger:
                        self._logger.info("no walk operator has been set. A new one will be built")
                    self.create_walk_operator(coord_format=CoordinateMultiplier, storage_level=storage_level)

            if self._num_particles > 1 and phase and self._interaction_operator is None:
                if self._logger:
                    self._logger.info("no interaction operator has been set. A new one will be built")
                self.create_interaction_operator(phase, coord_format=CoordinateMultiplier, storage_level=storage_level)

            t1 = datetime.now()

            if self._logger:
                self._logger.info("starting the walk...")

            for i in range(1, steps + 1, 1):
                if self._mesh.broken_links:
                    self.destroy_shift_operator()
                    self.destroy_walk_operator()
                    self.create_walk_operator(coord_format=CoordinateMultiplier, storage_level=storage_level)

                t_tmp = datetime.now()

                result_tmp = result

                if self._num_particles == 1:
                    result_tmp = self._walk_operator.multiply(result_tmp)
                else:
                    if self._interaction_operator is not None:
                        result_tmp = self._interaction_operator.multiply(result_tmp)

                    for wo in self._walk_operator:
                        result_tmp = wo.multiply(result_tmp)

                result_tmp.materialize(storage_level)
                result.unpersist()

                result = result_tmp

                if self._profiler:
                    self._profiler.profile_resources(app_id)
                    self._profiler.profile_executors(app_id)

                    info = self._profiler.profile_state(
                        'systemState{}'.format(i), result, (datetime.now() - t_tmp).total_seconds()
                    )

                    if self._logger:
                        self._logger.info("step was done in {}s".format(info['buildingTime']))
                        self._logger.info(
                            "system state of step {} is consuming {} bytes in memory and {} bytes in disk".format(
                                i, info['memoryUsed'], info['diskUsed']
                            )
                        )

                if self._logger:
                    self._profiler.log_rdd(app_id=app_id)

            if self._logger:
                self._logger.info("walk was done in {}s".format((datetime.now() - t1).total_seconds()))

            t1 = datetime.now()

            if self._logger:
                self._logger.debug("checking if the final state is unitary...")

            if not result.is_unitary():
                if self._logger:
                    self._logger.error("the final state is not unitary")
                raise ValueError("the final state is not unitary")

            if self._logger:
                self._logger.debug("unitarity check was done in {}s".format((datetime.now() - t1).total_seconds()))

        if self._profiler:
            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_state('finalState', result, 0.0)

            if self._logger:
                self._logger.info(
                    "final state is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

        return result
