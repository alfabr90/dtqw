from .utils.utils import *


class DiscreteTimeQuantumWalk:
    DTQW1D_WALK_LINE = 2 ** 0
    DTQW1D_WALK_CYCLE = 2 ** 1
    DTQW1D_WALK_SEGMENT = 2 ** 2

    DTQW2D_WALK_LATTICE = 2 ** 8
    DTQW2D_WALK_TORUS = 2 ** 9
    DTQW2D_WALK_BOX = 2 ** 10
    DTQW2D_MESH_DIAGONAL = 2 ** 30
    DTQW2D_MESH_NATURAL = 2 ** 31

    def __init__(self, spark_context, coin, size,
                 walk_type=DTQW1D_WALK_LINE, mesh_type=None,
                 num_particles=1, min_partitions=8, save_mode=SAVE_MODE_MEMORY,
                 storage_level=StorageLevel.MEMORY_ONLY, debug_file='./LOG.txt'):
        self.__spark_context = spark_context

        if num_particles < 1:
            raise Exception("There must be at least one particle!")

        self.__min_partitions = min_partitions
        self.__save_mode = save_mode
        self.__storage_level = storage_level

        self.__execution_times = DiscreteTimeQuantumWalk.__build_execution_times()
        self.__memory_usage = DiscreteTimeQuantumWalk.__build_memory_usage()

        self.__num_particles = num_particles
        self.__coin = coin
        self.__walk_type, self.__mesh_type, self.__num_dimensions = self.__validate_walk(walk_type, mesh_type)
        self.__size, self.__space = self.__build_space(size)

        self.__coin_operator = None
        self.__shift_operator = None
        self.__unitary_operator = None
        self.__interaction_operator = None
        self.__multiparticles_unitary_operator = None
        self.__walk_operator = None

        self.__steps = 0

        self.__debug_file = debug_file

    @property
    def spark_context(self):
        return self.__spark_context

    @property
    def walk_type(self):
        return self.__walk_type

    @property
    def mesh_type(self):
        return self.__mesh_type

    @property
    def num_dimensions(self):
        return self.__num_dimensions

    @property
    def coin(self):
        return self.__coin

    @property
    def size(self):
        return self.__size

    @property
    def space(self):
        return self.__space

    @property
    def num_particles(self):
        return self.__num_particles

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

    @shift_operator.setter
    def shift_operator(self, so):
        if so is None:
            self.__shift_operator = so
        elif sp.isspmatrix(so):
            if not sp.isspmatrix_csc(so):
                self.__shift_operator = so.tocsc()
            else:
                self.__shift_operator = so
        elif type(self.__shift_operator) == str:
            if os.path.exists(so):
                self.__shift_operator = so
            else:
                raise Exception("The path of shift operator does not exist!")
        else:
            raise Exception("Unsupported type for shift operator!")

    @unitary_operator.setter
    def unitary_operator(self, uo):
        if self.__unitary_operator is None:
            self.__unitary_operator = uo
        elif sp.isspmatrix(uo):
            if not sp.isspmatrix_csc(uo):
                self.__unitary_operator = uo.tocsc()
            else:
                self.__unitary_operator = uo
        elif type(uo) == str:
            if os.path.exists(uo):
                self.__unitary_operator = uo
            else:
                raise Exception("The path of unitary operator does not exist!")
        else:
            raise Exception("Unsupported type for unitary operator!")

    @interaction_operator.setter
    def interaction_operator(self, io):
        if io is None:
            self.__interaction_operator = io
        elif sp.isspmatrix(io):
            if not sp.isspmatrix_csc(io):
                self.__interaction_operator = io.tocsc()
            else:
                self.__interaction_operator = io
        elif type(self.__interaction_operator) == str:
            if os.path.exists(io):
                self.__unitary_operator = io
            else:
                raise Exception("The path of interaction operator does not exist!")
        else:
            raise Exception("Unsupported type for interaction operator!")

    @multiparticles_unitary_operator.setter
    def multiparticles_unitary_operator(self, mu):
        if mu is None:
            self.__multiparticles_unitary_operator = mu
        elif sp.isspmatrix(mu):
            if not sp.isspmatrix_csc(mu):
                self.__multiparticles_unitary_operator = mu.tocsc()
            else:
                self.__multiparticles_unitary_operator = mu
        elif type(self.__interaction_operator) == str:
            if os.path.exists(mu):
                self.__multiparticles_unitary_operator = mu
            else:
                raise Exception("The path of multiparticles unitary operator does not exist!")
        else:
            raise Exception("Unsupported type for multiparticles unitary operator!")

    @walk_operator.setter
    def walk_operator(self, w):
        if w is None:
            self.__walk_operator = w
        elif sp.isspmatrix(w):
            if not sp.isspmatrix_csc(w):
                self.__walk_operator = w.tocsc()
            else:
                self.__walk_operator = w
        elif type(self.__interaction_operator) == str:
            if os.path.exists(w):
                self.__walk_operator = w
            else:
                raise Exception("The path of walk operator does not exist!")
        else:
            raise Exception("Unsupported type for walk operator!")

    @staticmethod
    def __build_execution_times():
        return {
            'coin_operator': 0.0,
            'shift_operator_tmp': 0.0,
            'shift_operator': 0.0,
            'unitary_operator_tmp': 0.0,
            'unitary_operator': 0.0,
            'interaction_operator_tmp': 0.0,
            'interaction_operator': 0.0,
            'multiparticle_unitary_operator_tmp': 0.0,
            'multiparticles_unitary_operator': 0.0,
            'walk_operator': 0.0,
            'walk': 0.0,
            'export_state': 0.0,
            'measurement_tmp': 0.0,
            'full_measurement': 0.0,
            'filtered_measurement': 0.0,
            'partial_measurement': 0.0,
            'export_pdf': 0.0,
            'filter_pdf': 0.0,
            'export_plot': 0.0
        }

    @staticmethod
    def __build_memory_usage():
        return {
            'coin_operator': 0,
            'shift_operator_tmp': 0,
            'shift_operator': 0,
            'unitary_operator_tmp': 0,
            'unitary_operator': 0,
            'interaction_operator_tmp': 0,
            'interaction_operator': 0,
            'multiparticle_unitary_operator_tmp': 0,
            'multiparticles_unitary_operator': 0,
            'walk_operator': 0,
            'state': 0,
            'pdf': 0
        }

    def __validate_walk(self, walk, mesh=None):
        if walk == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE \
                or walk == DiscreteTimeQuantumWalk.DTQW1D_WALK_SEGMENT \
                or walk == DiscreteTimeQuantumWalk.DTQW1D_WALK_CYCLE:
            if self.__coin.shape[0] == 2 and self.__coin.shape[1] == 2:
                return walk, mesh, 1
            else:
                raise Exception("Incompatible coin size!")
        elif walk == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE \
                or walk == DiscreteTimeQuantumWalk.DTQW2D_WALK_BOX \
                or walk == DiscreteTimeQuantumWalk.DTQW2D_WALK_TORUS:
            if mesh == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL \
                    or mesh == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                if self.__coin.shape[0] == 4 and self.__coin.shape[1] == 4:
                    return walk, mesh, 2
                else:
                    raise Exception("Incompatible coin size!")
            else:
                raise Exception("Unsupported mesh type!")
        else:
            raise Exception("Walk type not implemented!")

    def __build_space(self, size):
        if self.__num_dimensions == 1:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                return size * 2 + 1, sp.identity(size * 2 + 1, format='coo')
            else:
                return size, sp.identity(size, format='coo')
        elif self.__num_dimensions == 2:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                return (
                    (size[0] * 2 + 1, size[1] * 2 + 1),
                    (sp.identity(size[0] * 2 + 1, format='coo'), sp.identity(size[1] * 2 + 1, format='coo'))
                )
            else:
                return size, (sp.identity(size[0], format='coo'), sp.identity(size[1], format='coo'))

    def __build_coin_operator(self):
        t1 = datetime.now()

        if self.__save_mode == SAVE_MODE_DISK:
            if self.__num_dimensions == 1:
                shape = self.__space.shape

                result = parallel_kron(
                    self.__coin,
                    self.__space,
                    self.__spark_context,
                    (self.__coin.shape[0] * shape[0], self.__coin.shape[0] * shape[1]),
                    min_partitions=self.__min_partitions
                )
            elif self.__num_dimensions == 2:
                shape = (
                    self.__space[0].shape[0] * self.__space[1].shape[0],
                    self.__space[0].shape[1] * self.__space[1].shape[1]
                )

                result = parallel_kron(
                    self.__coin,
                    parallel_kron(
                        self.__space[0],
                        self.__space[1],
                        self.__spark_context,
                        shape,
                        min_partitions=self.__min_partitions
                    ),
                    self.__spark_context,
                    (self.__coin.shape[0] * shape[0], self.__coin.shape[0] * shape[1]),
                    min_partitions=self.__min_partitions
                )
        else:
            if self.__num_dimensions == 1:
                result = sp.kron(self.__coin, self.__space, 'csr')
            elif self.__num_dimensions == 2:
                result = sp.kron(self.__coin, sp.kron(self.__space[0], self.__space[1]), 'csr')

        t2 = datetime.now()
        self.__execution_times['coin_operator'] = (t2 - t1).total_seconds()
        if self.__save_mode == SAVE_MODE_DISK:
            self.__memory_usage['coin_operator'] = size_of_tmp_path(result)
        else:
            self.__memory_usage['coin_operator'] = get_size_of(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                if self.__num_dimensions == 1:
                    f.write("Shape of coin operator: {}\n".format(
                        (self.__coin.shape[0] * self.__space.shape[0], self.__coin.shape[1] * self.__space.shape[1]))
                    )
                elif self.__num_dimensions == 2:
                    f.write("Shape of coin operator: {}\n".format(
                        (
                            self.__coin.shape[0] * self.__space[0].shape[0] * self.__space[1].shape[0],
                            self.__coin.shape[1] * self.__space[0].shape[1] * self.__space[1].shape[1]
                        )
                    ))
                f.write("Coin operator in {}s\n".format(self.__execution_times['coin_operator']))
                f.write("Coin operator is consuming {} bytes\n".format(self.__memory_usage['coin_operator']))
                if self.__save_mode == SAVE_MODE_DISK:
                    f.write("Coin operator path: {}\n".format(result))

        return result

    def __build_shift_operator(self):
        t1 = datetime.now()

        path = get_tmp_path()

        cs = coin_space(2)

        if self.__num_dimensions == 1:
            data_size = self.__coin.shape[0] * self.__size * get_size_of(0)
            buffer_size = get_buffer_size(data_size)

            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                with open(path, 'w', buffer_size) as f:
                    for i in range(cs.shape[0]):
                        l = (-1) ** i
                        for x in range(self.__size):
                            f.write("{} {} {} {}\n".format(i, i, (x + l) % self.__size, x))
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_SEGMENT:
                with open(path, 'w', buffer_size) as f:
                    for i in range(cs.shape[0]):
                        l = (-1) ** i
                        for x in range(self.__size):
                            if x + l >= self.__size or x + l < 0:
                                bl = 0
                            else:
                                bl = l
                            f.write("{} {} {} {}\n".format(i + bl, 1 - i, x + bl, x))
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_CYCLE:
                with open(path, 'w', buffer_size) as f:
                    for i in range(cs.shape[0]):
                        l = (-1) ** i
                        for x in range(self.__size):
                            f.write("{} {} {} {}\n".format(i, i, (x + l) % self.__size, x))
        elif self.__num_dimensions == 2:
            data_size = self.__coin.shape[0] * self.__size[0] * self.__size[1] * get_size_of(0)
            buffer_size = get_buffer_size(data_size)

            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                # |x+(-1)**i,y+(-1)**j><x,y|
                if self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l1 = (-1) ** i
                            for j in range(cs.shape[0]):
                                l2 = (-1) ** j
                                for x in range(self.__space[0].shape[0]):
                                    for y in range(self.__space[1].shape[0]):
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i, j, i, j,
                                                (x + l1) % self.__space[0].shape[0],
                                                (y + l2) % self.__space[1].shape[0],
                                                x, y
                                            )
                                        )
                elif self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l = (-1) ** i
                            for j in range(cs.shape[0]):
                                delta = int(braket(cs[:, i], cs[:, j]))
                                for x in range(self.__space[0].shape[0]):
                                    for y in range(self.__space[1].shape[0]):
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i, j, i, j,
                                                (x + l * (1 - delta)) % self.__space[0].shape[0],
                                                (y + l * delta) % self.__space[1].shape[0],
                                                x, y
                                            )
                                        )
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_BOX:
                if self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l1 = (-1) ** i
                            for j in range(cs.shape[0]):
                                l2 = (-1) ** j
                                for x in range(self.__size[0]):
                                    for y in range(self.__size[1]):
                                        if x + l1 >= self.__size[0] or x + l1 < 0 \
                                                or y + l2 >= self.__size[1] or y + l2 < 0:
                                            bl1 = 0
                                            bl2 = 0
                                        else:
                                            bl1 = l1
                                            bl2 = l2
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i + bl1, j + bl2, 1 - i, 1 - j, x + bl1, y + bl2, x, y
                                            )
                                        )
                elif self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l = (-1) ** i
                            for j in range(cs.shape[0]):
                                delta = int(braket(cs[:, i], cs[:, j]))
                                for x in range(self.__size[0]):
                                    for y in range(self.__size[1]):
                                        pos1 = x + l * (1 - delta)
                                        pos2 = y + l * delta

                                        if pos1 >= self.__size[0] or pos1 < 0 or pos2 >= self.__size[1] or pos2 < 0:
                                            bl = 0
                                        else:
                                            bl = l
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i + bl, abs(j + bl) % cs.shape[0], 1 - i, 1 - j,
                                                x + bl * (1 - delta), y + bl * delta, x, y
                                            )
                                        )
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_TORUS:
                if self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l1 = (-1) ** i
                            for j in range(cs.shape[0]):
                                l2 = (-1) ** j
                                for x in range(self.__space[0].shape[0]):
                                    for y in range(self.__space[1].shape[0]):
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i, j, i, j,
                                                (x + l1) % self.__space[0].shape[0],
                                                (y + l2) % self.__space[1].shape[0],
                                                x, y
                                            )
                                        )
                elif self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                    with open(path, 'w', buffer_size) as f:
                        for i in range(cs.shape[0]):
                            l = (-1) ** i
                            for j in range(cs.shape[0]):
                                delta = int(braket(cs[:, i], cs[:, j]))
                                for x in range(self.__space[0].shape[0]):
                                    for y in range(self.__space[1].shape[0]):
                                        f.write(
                                            "{} {} {} {} {} {} {} {}\n".format(
                                                i, j, i, j,
                                                (x + l * (1 - delta)) % self.__space[0].shape[0],
                                                (y + l * delta) % self.__space[1].shape[0],
                                                x, y
                                            )
                                        )

        self.__execution_times['shift_operator_tmp'] = (datetime.now() - t1).total_seconds()
        self.__memory_usage['shift_operator_tmp'] = size_of_tmp_path(path)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Buffer size: {}\n".format(buffer_size))
                f.write("Shift operator temp in {}s\n".format(self.__execution_times['shift_operator_tmp']))
                f.write("Shift operator temp consumed {} bytes in disk\n".format(self.__memory_usage['shift_operator_tmp']))

        cs = broadcast(self.__spark_context, cs)

        if self.__num_dimensions == 1:
            s_csc = broadcast(self.__spark_context, self.__space.tocsc())
            s_csr = broadcast(self.__spark_context, self.__space.tocsr())
        elif self.__num_dimensions == 2:
            s_csc = broadcast(self.__spark_context, (self.__space[0].tocsc(), self.__space[1].tocsc()))
            s_csr = broadcast(self.__spark_context, (self.__space[0].tocsr(), self.__space[1].tocsr()))

        if self.__save_mode == SAVE_MODE_DISK:
            path2 = get_tmp_path()

            if self.__num_dimensions == 1:
                def __map(m):
                    a = [int(i) for i in m.split()]

                    ind = sp.find(sp.kron(
                        cs.value[:, a[0]] * cs.value[a[1], :],
                        s_csc.value[:, a[2]] * s_csr.value[a[3], :]
                    ))

                    k = []

                    for i in range(len(ind[2])):
                        k.append("{} {} {}".format(ind[0][i], ind[1][i], ind[2][i]))

                    del ind

                    return "\n".join(k)
            elif self.__num_dimensions == 2:
                def __map(m):
                    a = [int(i) for i in m.split()]

                    ind = sp.find(sp.kron(
                        sp.kron(cs.value[:, a[0]], cs.value[:, a[1]]) *
                        sp.kron(cs.value[a[2], :], cs.value[a[3], :]),
                        sp.kron(s_csc.value[0][:, a[4]], s_csc.value[1][:, a[5]]) *
                        sp.kron(s_csr.value[0][a[6], :], s_csr.value[1][a[7], :])
                    ))

                    k = []

                    for i in range(len(ind[2])):
                        k.append("{} {} {}".format(ind[0][i], ind[1][i], ind[2][i]))

                    del ind

                    return "\n".join(k)

            self.__spark_context.textFile(
                path, minPartitions=self.__min_partitions
            ).map(
                __map
            ).saveAsTextFile(path2)

            result = path2
        else:
            if self.__num_dimensions == 1:
                def __map(m):
                    a = [int(i) for i in m.split()]
                    return sp.kron(
                        cs.value[:, a[0]] * cs.value[a[1], :],
                        s_csc.value[:, a[2]] * s_csr.value[a[3], :],
                        'csc'
                    )
            elif self.__num_dimensions == 2:
                def __map(m):
                    a = [int(i) for i in m.split()]
                    return sp.kron(
                        sp.kron(cs.value[:, a[0]], cs.value[:, a[1]]) *
                        sp.kron(cs.value[a[2], :], cs.value[a[3], :]),
                        sp.kron(s_csc.value[0][:, a[4]], s_csc.value[1][:, a[5]]) *
                        sp.kron(s_csr.value[0][a[6], :], s_csr.value[1][a[7], :]),
                        'csc'
                    )

            result = self.__spark_context.textFile(
                path, minPartitions=self.__min_partitions
            ).map(
                __map
            ).reduce(
                lambda a, b: a + b
            )

        cs.unpersist()
        s_csc.unpersist()
        s_csr.unpersist()

        remove_tmp_path(path)

        t2 = datetime.now()
        self.__execution_times['shift_operator'] = (t2 - t1).total_seconds()
        if self.__save_mode == SAVE_MODE_DISK:
            self.__memory_usage['shift_operator'] = size_of_tmp_path(result)
        else:
            self.__memory_usage['shift_operator'] = get_size_of(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                if self.__num_dimensions == 1:
                    f.write("Shape of shift operator: {}\n".format(
                        (self.__coin.shape[0] * self.__space.shape[0], self.__coin.shape[1] * self.__space.shape[1]))
                    )
                elif self.__num_dimensions == 2:
                    f.write("Shape of shift operator: {}\n".format(
                        (
                            self.__coin.shape[0] * self.__space[0].shape[0] * self.__space[1].shape[0],
                            self.__coin.shape[1] * self.__space[0].shape[1] * self.__space[1].shape[1]
                        )
                    ))
                f.write("Shift operator in {}s\n".format(self.__execution_times['shift_operator']))
                if self.__save_mode == SAVE_MODE_DISK:
                    f.write("Shift operator path: {}\n".format(result))

        return result

    def __build_unitary_operator(self):
        t1 = datetime.now()

        if self.__shift_operator is None:
            raise Exception("The shift operator has not been built!")

        result = mat_mat_product(
            self.__shift_operator,
            self.__coin_operator,
            self.__spark_context,
            min_partitions=self.__min_partitions,
            save_mode=self.__save_mode
        )

        t2 = datetime.now()
        self.__execution_times['unitary_operator'] = (t2 - t1).total_seconds()
        if self.__save_mode == SAVE_MODE_DISK:
            self.__memory_usage['unitary_operator'] = size_of_tmp_path(result)
        else:
            self.__memory_usage['unitary_operator'] = get_size_of(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                if self.__num_dimensions == 1:
                    f.write("Shape of unitary operator: {}\n".format(
                        (self.__coin.shape[0] * self.__space.shape[0], self.__coin.shape[1] * self.__space.shape[1]))
                    )
                elif self.__num_dimensions == 2:
                    f.write("Shape of unitary operator: {}\n".format(
                        (
                            self.__coin.shape[0] * self.__space[0].shape[0] * self.__space[1].shape[0],
                            self.__coin.shape[1] * self.__space[0].shape[1] * self.__space[1].shape[1]
                        )
                    ))
                f.write("Unitary operator in {}s\n".format(self.__execution_times['unitary_operator']))
                if self.__save_mode == SAVE_MODE_DISK:
                    f.write("Unitary operator path: {}\n".format(result))

        return result

    def __build_interaction_operator(self, phase):
        t1 = datetime.now()

        ndim = self.__num_dimensions
        num_p = self.__num_particles
        cp = cmath.exp(phase * (0.0+1.0j))
        cs = coin_space(2)

        path = get_tmp_path()

        if self.__num_dimensions == 1:
            size = self.__size
            cs_size = cs.shape[0] * self.__size
            shape = (cs_size ** num_p, cs_size ** num_p)

            def __map(m):
                a = []
                for p in range(num_p):
                    a.append(int(m / (cs_size ** (num_p - 1 - p))) % size)
                for i in range(num_p):
                    if a[0] != a[i]:
                        return "{} {} {}".format(m, m, 1)
                return "{} {} {}".format(m, m, cp)

        elif self.__num_dimensions == 2:
            ind = self.__num_dimensions * self.__num_particles
            size_x = self.__size[0]
            size_y = self.__size[1]
            cs_size_x = cs.shape[0] * self.__size[0]
            cs_size_y = cs.shape[0] * self.__size[1]
            shape = ((cs_size_x * cs_size_y) ** num_p, (cs_size_x * cs_size_y) ** num_p)

            def __map(m):
                a = []
                for p in range(num_p):
                    a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                    a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                for i in range(0, ind, ndim):
                    if a[0] != a[i] or a[1] != a[i + 1]:
                        return "{} {} {}".format(m, m, 1)
                return "{} {} {}".format(m, m, cp)

        self.__spark_context.range(
            shape[0]
        ).map(
            __map
        ).saveAsTextFile(path)

        result = path

        t2 = datetime.now()
        self.__execution_times['interaction_operator'] = (t2 - t1).total_seconds()
        self.__memory_usage['interaction_operator'] = size_of_tmp_path(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Shape of interaction operator: {}\n".format(shape))
                f.write("Interaction operator in {}s\n".format(self.__execution_times['interaction_operator']))
                f.write("Interaction operator path: {}\n".format(result))

        return result

    def __build_multiparticles_unitary_operator(self):
        t1 = datetime.now()

        num_p = self.__num_particles

        if self.__num_dimensions == 1:
            shape = (self.__coin.shape[0] * self.__space.shape[0], self.__coin.shape[1] * self.__space.shape[1])
        elif self.__num_dimensions == 2:
            shape = (
                self.__coin.shape[0] * self.__space[0].shape[0] * self.__space[1].shape[0],
                self.__coin.shape[1] * self.__space[0].shape[1] * self.__space[1].shape[1]
            )

        result = self.__unitary_operator

        for p in range(num_p - 1):
            result = parallel_kron(
                result,
                self.__unitary_operator,
                self.__spark_context,
                shape,
                min_partitions=self.__min_partitions
            )

        t2 = datetime.now()
        self.__execution_times['multiparticles_unitary_operator'] = (t2 - t1).total_seconds()
        self.__memory_usage['multiparticles_unitary_operator'] = size_of_tmp_path(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                if self.__num_dimensions == 1:
                    f.write("Shape of multiparticle unitary operator: {}\n".format(
                        (
                            (self.__coin.shape[0] * self.__space.shape[0]) ** self.__num_particles,
                            (self.__coin.shape[1] * self.__space.shape[1]) ** self.__num_particles
                        )
                    ))
                elif self.__num_dimensions == 2:
                    f.write("Shape of multiparticle unitary operator: {}\n".format(
                        (
                            (self.__coin.shape[0] * self.__space[0].shape[0] * self.__space[1].shape[0])
                            ** self.__num_particles,
                            (self.__coin.shape[1] * self.__space[0].shape[1] * self.__space[1].shape[1])
                            ** self.__num_particles
                        )
                    ))
                f.write("Multiparticles unitary operator in {}s\n".format(
                    self.__execution_times['multiparticles_unitary_operator'])
                )
                f.write("Multiparticles unitary operator path: {}\n".format(result))

        return result

    def __build_operators(self, collision_phase=None):
        if self.__walk_operator is None:
            if self.__coin_operator is None:
                self.__coin_operator = self.__build_coin_operator()

            if self.__num_particles > 1:
                if self.__multiparticles_unitary_operator is None:
                    if self.__unitary_operator is None:
                        if self.__shift_operator is None:
                            self.__shift_operator = self.__build_shift_operator()
                        elif sp.isspmatrix(self.__shift_operator):
                            self.__memory_usage['shift_operator'] = get_size_of(self.__shift_operator)
                        elif type(self.__shift_operator) == str:
                            self.__memory_usage['shift_operator'] = size_of_tmp_path(self.__shift_operator)

                        self.__unitary_operator = self.__build_unitary_operator()
                    elif sp.isspmatrix(self.__unitary_operator):
                        self.__memory_usage['unitary_operator'] = get_size_of(self.__unitary_operator)
                    elif type(self.__unitary_operator) == str:
                        self.__memory_usage['unitary_operator'] = size_of_tmp_path(self.__unitary_operator)

                    if DEBUG_MODE:
                        with open(self.__debug_file, 'a+') as f:
                            f.write(
                                "Shift operator is consuming {} bytes\n".format(
                                    self.__memory_usage['shift_operator']
                                )
                            )
                            f.write(
                                "Unitary operator is consuming {} bytes\n".format(
                                    self.__memory_usage['unitary_operator']
                                )
                            )

                    self.__multiparticles_unitary_operator = self.__build_multiparticles_unitary_operator()
                elif sp.isspmatrix(self.__multiparticles_unitary_operator):
                    self.__memory_usage['multiparticles_unitary_operator'] = get_size_of(
                        self.__multiparticles_unitary_operator
                    )
                elif type(self.__multiparticles_unitary_operator) == str:
                    self.__memory_usage['multiparticles_unitary_operator'] = size_of_tmp_path(
                        self.__multiparticles_unitary_operator
                    )

                if DEBUG_MODE:
                    with open(self.__debug_file, 'a+') as f:
                        f.write(
                            "Multiparticles unitary operator is consuming {} bytes\n".format(
                                self.__memory_usage['multiparticles_unitary_operator']
                            )
                        )

                if self.__interaction_operator is None and collision_phase is not None:
                    self.__interaction_operator = self.__build_interaction_operator(collision_phase)
                elif sp.isspmatrix(self.__interaction_operator):
                    self.__memory_usage['interaction_operator'] = get_size_of(self.__interaction_operator)
                elif type(self.__interaction_operator) == str:
                    self.__memory_usage['interaction_operator'] = size_of_tmp_path(self.__interaction_operator)

                if self.__interaction_operator is None:
                    self.__walk_operator = self.__multiparticles_unitary_operator
                else:
                    if DEBUG_MODE:
                        with open(self.__debug_file, 'a+') as f:
                            f.write(
                                "Interaction operator is consuming {} bytes\n".format(
                                    self.__memory_usage['interaction_operator']
                                )
                            )
                    
                    t1 = datetime.now()

                    self.__walk_operator = mat_mat_product(
                        self.__multiparticles_unitary_operator,
                        self.__interaction_operator,
                        self.__spark_context,
                        min_partitions=self.__min_partitions,
                        save_mode=SAVE_MODE_DISK
                    )
                    
                    self.__execution_times['walk_operator'] = (datetime.now() - t1).total_seconds()
                    self.__memory_usage['walk_operator'] = size_of_tmp_path(self.__walk_operator)

                    if DEBUG_MODE:
                        with open(self.__debug_file, 'a+') as f:
                            f.write(
                                "Walk operator in {}s\n".format(
                                    self.__execution_times['walk_operator']
                                )
                            )
                            f.write("Walk operator path: {}\n".format(self.__walk_operator))
                            f.write(
                                "Walk operator is consuming {} bytes in disk\n".format(
                                    self.__memory_usage['walk_operator']
                                )
                            )
            else:
                if self.__unitary_operator is None:
                    if self.__shift_operator is None:
                        self.__shift_operator = self.__build_shift_operator()
                    elif sp.isspmatrix(self.__shift_operator):
                        self.__memory_usage['shift_operator'] = get_size_of(self.__shift_operator)
                    elif type(self.__shift_operator) == str:
                        self.__memory_usage['shift_operator'] = size_of_tmp_path(self.__shift_operator)

                    self.__unitary_operator = self.__build_unitary_operator()
                elif sp.isspmatrix(self.__unitary_operator):
                    self.__memory_usage['unitary_operator'] = get_size_of(self.__unitary_operator)
                elif type(self.__unitary_operator) == str:
                    self.__memory_usage['unitary_operator'] = size_of_tmp_path(self.__unitary_operator)

                if DEBUG_MODE:
                    with open(self.__debug_file, 'a+') as f:
                        f.write("Shift operator is consuming {} bytes\n".format(self.__memory_usage['shift_operator']))
                        f.write(
                            "Unitary operator is consuming {} bytes\n".format(
                                self.__memory_usage['unitary_operator']
                            )
                        )

                self.__walk_operator = self.__unitary_operator
        elif sp.isspmatrix(self.__walk_operator):
            self.__memory_usage['walk_operator'] = get_size_of(self.__walk_operator)
        elif type(self.__walk_operator) == str:
            self.__memory_usage['walk_operator'] = size_of_tmp_path(self.__walk_operator)

    def plot_title(self):
        if self.__num_dimensions == 1:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                walk = "Line"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_SEGMENT:
                walk = "Segment"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_CYCLE:
                walk = "Cycle"

            if self.__num_particles == 1:
                particles = str(self.__num_particles) + " Particle on a "
            else:
                particles = str(self.__num_particles) + " Particles on a "

            return "Quantum Walk with " + particles + walk
        elif self.__num_dimensions == 2:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                walk = "Lattice"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_BOX:
                walk = "Box"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_TORUS:
                walk = "Torus"

            if self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL:
                mesh = "Diagonal"
            elif self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                mesh = "Natural"

            if self.__num_particles == 1:
                particles = str(self.__num_particles) + " Particle on a "
            else:
                particles = str(self.__num_particles) + " Particles on a "

            return "Quantum Walk with " + particles + mesh + " " + walk

    def output_filename(self):
        if self.__num_dimensions == 1:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                walk = "LINE"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_SEGMENT:
                walk = "SEGMENT"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_CYCLE:
                walk = "CYCLE"

            size = str(self.__size)

            return "DTQW1D_{}_{}_{}_{}_{}".format(
                walk, size, self.__steps, self.__num_particles, self.__min_partitions
            )
        elif self.__num_dimensions == 2:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                walk = "LATTICE"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_BOX:
                walk = "BOX"
            elif self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_TORUS:
                walk = "TORUS"

            if self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_DIAGONAL:
                mesh = "DIAGONAL"
            elif self.__mesh_type == DiscreteTimeQuantumWalk.DTQW2D_MESH_NATURAL:
                mesh = "NATURAL"

            size = str(self.__size[0]) + "-" + str(self.__size[1])

            return "DTQW2D_{}_{}_{}_{}_{}_{}".format(
                walk, mesh, size, self.__steps, self.__num_particles, self.__min_partitions
            )

    def clear_operators(self):
        if self.__coin_operator is not None:
            if sp.isspmatrix(self.__coin_operator):
                self.__coin_operator = None
            elif type(self.__coin_operator) == str:
                remove_tmp_path(self.__coin_operator)

        if self.__shift_operator is not None:
            if sp.isspmatrix(self.__shift_operator):
                self.__shift_operator = None
            elif type(self.__shift_operator) == str:
                remove_tmp_path(self.__shift_operator)

        if self.__unitary_operator is not None:
            if sp.isspmatrix(self.__unitary_operator):
                self.__unitary_operator = None
            elif type(self.__unitary_operator) == str:
                remove_tmp_path(self.__unitary_operator)

        if self.__interaction_operator is not None:
            if sp.isspmatrix(self.__interaction_operator):
                self.__interaction_operator = None
            elif type(self.__interaction_operator) == str:
                remove_tmp_path(self.__interaction_operator)

        if self.__multiparticles_unitary_operator is not None:
            if sp.isspmatrix(self.__multiparticles_unitary_operator):
                self.__multiparticles_unitary_operator = None
            elif type(self.__multiparticles_unitary_operator) == str:
                remove_tmp_path(self.__multiparticles_unitary_operator)

        if self.__walk_operator is not None:
            if sp.isspmatrix(self.__walk_operator):
                self.__walk_operator = None
            elif type(self.__walk_operator) == str:
                remove_tmp_path(self.__walk_operator)

    def walk(self, steps, initial_state, collision_phase=None):
        if steps < 0:
            raise Exception("The number of steps cannot be negative!")

        if self.__num_dimensions == 1:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                if steps > int((self.__size - 1) / 2):
                    raise Exception("The number of steps cannot be greater than the size of the lattice!")
        elif self.__num_dimensions == 2:
            if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                if steps > int((self.__size[0] - 1) / 2) or steps > int((self.__size[1] - 1) / 2):
                    raise Exception("The number of steps cannot be greater than the size of the lattice!")

        self.__steps = steps

        if sp.isspmatrix(initial_state):
            if not sp.isspmatrix_csc(initial_state):
                result = initial_state.tocsc()
            else:
                result = initial_state
        elif isinstance(initial_state, np.ndarray):
            result = initial_state
        elif type(initial_state) == str:
            if os.path.exists(initial_state):
                result = initial_state
            else:
                raise FileNotFoundError
        else:
            raise Exception("Unsupported type for initial state!")

        if isunitary(result, self.__spark_context) is False:
            raise Exception("The initial state is not unitary!")

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Steps: {}\n".format(self.__steps))
                f.write("Space size: {}\n".format(self.__size))
                f.write("Nº of partitions: {}\n".format(self.__min_partitions))
                f.write("Nº of particles: {}\n".format(self.__num_particles))
                if self.__num_particles > 1:
                    f.write("Collision phase: {}\n".format(collision_phase))

        if self.__steps > 0:
            self.__build_operators(collision_phase)

            t1 = datetime.now()

            wo = self.__walk_operator
            
            unpersist = False

            if not sp.isspmatrix(wo) or not isinstance(wo, np.ndarray):
                def __map(m):
                    a = m.split()
                    return int(a[0]), int(a[1]), complex(a[2])

                wo = self.__spark_context.textFile(
                    self.__walk_operator, minPartitions=self.__min_partitions
                ).map(
                    __map
                ).filter(
                    lambda m: m[2] != (0+0j)
                ).persist(self.__storage_level)
                
                unpersist = True

            # if not sp.isspmatrix(self.__walk_operator) and sp.isspmatrix(result):
                # result = sparse_to_disk(result, min_partitions=self.__min_partitions)

            t_tmp = datetime.now()

            result = mat_vec_product(
                wo,
                result,
                self.__spark_context,
                min_partitions=self.__min_partitions,
                save_mode=self.__save_mode
            )

            if DEBUG_MODE:
                with open(self.__debug_file, 'a+') as f:
                    f.write("Step in {}s\n".format((datetime.now() - t_tmp).total_seconds()))
                    '''
                    if self.__save_mode == SAVE_MODE_MEMORY:
                        f.write("Nonzero elements in state: {}\n".format(result.nnz))
                    if self.__save_mode == SAVE_MODE_MEMORY:
                        f.write("Result is consuming {} bytes\n".format(get_size_of(result)))
                    elif self.__save_mode == SAVE_MODE_DISK:
                        f.write("Result is consuming {} bytes\n".format(size_of_tmp_path(result)))
                    '''

            if isunitary(result, self.__spark_context) is False:
                raise Exception("The state is not unitary!")

            for i in range(self.__steps - 1):
                t_tmp = datetime.now()

                result_tmp = mat_vec_product(
                    wo,
                    result,
                    self.__spark_context,
                    min_partitions=self.__min_partitions,
                    save_mode=self.__save_mode
                )

                if sp.isspmatrix(result) or isinstance(result, np.ndarray):
                    del result
                elif type(result) == str:
                    remove_tmp_path(result)

                result = result_tmp

                if DEBUG_MODE:
                    with open(self.__debug_file, 'a+') as f:
                        f.write("Step in {}s\n".format((datetime.now() - t_tmp).total_seconds()))
                        '''
                        if self.__save_mode == SAVE_MODE_MEMORY:
                            f.write("Nonzero elements in state: {}\n".format(result.nnz))
                        if self.__save_mode == SAVE_MODE_MEMORY:
                            f.write("Result is consuming {} bytes\n".format(get_size_of(result)))
                        elif self.__save_mode == SAVE_MODE_DISK:
                            f.write("Result is consuming {} bytes\n".format(size_of_tmp_path(result)))
                        '''

                if isunitary(result, self.__spark_context) is False:
                    raise Exception("The state is not unitary!")
            
            if unpersist:
                wo.unpersist()
            
            if self.__save_mode == SAVE_MODE_DISK:
                if sp.isspmatrix(result):
                    result = sparse_to_disk(result, min_partitions=self.__min_partitions)
                elif isinstance(result, np.ndarray):
                    result = dense_to_disk(result, min_partitions=self.__min_partitions)
                elif isinstance(result, RDD):
                    path = get_tmp_path()

                    result.map(
                        lambda m: "{} {} {}".format(m[0], m[1], m[2])
                    ).saveAsTextFile(path)

                    result = path
            else:
                if isinstance(result, RDD):
                    if self.__num_dimensions == 1:
                        shape = ((self.__coin.shape[0] * self.__size) ** self.__num_particles, 1)
                    elif self.__num_dimensions == 2:
                        shape = ((self.__coin.shape[0] * self.__size[0] * self.__size[1]) ** self.__num_particles, 1)

                    result = rdd_to_dense(result, shape)

            t2 = datetime.now()
            self.__execution_times['walk'] += (t2 - t1).total_seconds()
            if self.__save_mode == SAVE_MODE_DISK:
                self.__memory_usage['state'] = size_of_tmp_path(result)
            else:
                self.__memory_usage['state'] = get_size_of(result)

            if DEBUG_MODE:
                with open(self.__debug_file, 'a+') as f:
                    f.write("Walk in {}s\n".format(self.__execution_times['walk']))
                    f.write("State is consuming {} bytes\n".format(self.__memory_usage['state']))

        return result

    def __full_measurement(self, state, save_mode):
        t1 = datetime.now()

        ndim = self.__num_dimensions
        num_p = self.__num_particles
        ind = self.__num_dimensions * self.__num_particles
        cs = coin_space(2)

        path = get_tmp_path()

        if self.__num_dimensions == 1:
            dims = [self.__size for p in range(ind)]

            size = self.__size
            cs_size = cs.shape[0] * self.__size
            shape = (cs_size ** num_p, cs_size ** num_p)
        elif self.__num_dimensions == 2:
            dims = []

            for p in range(0, ind, ndim):
                dims.append(self.__size[0])
                dims.append(self.__size[1])

            size_x = self.__size[0]
            size_y = self.__size[1]
            cs_size_x = cs.shape[0] * self.__size[0]
            cs_size_y = cs.shape[0] * self.__size[1]
            shape = ((cs_size_x * cs_size_y) ** num_p, (cs_size_x * cs_size_y) ** num_p)

        if sp.isspmatrix(state) or isinstance(state, np.ndarray):
            if sp.isspmatrix(state):
                s = broadcast(self.__spark_context, state.toarray())
            else:
                s = broadcast(self.__spark_context, state)

            if self.__num_dimensions == 1:
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m / (cs_size ** (num_p - 1 - p))) % size)
                    a.append((abs(s.value[m, 0]) ** 2).real)
                    return " ".join([str(i) for i in a])
            elif self.__num_dimensions == 2:
                def __map(m):
                    a = []
                    for p in range(num_p):
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                        a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                    a.append((abs(s.value[m, 0]) ** 2).real)
                    return " ".join([str(i) for i in a])

            self.__spark_context.range(
                shape[0]
            ).filter(
                lambda m: s.value[m, 0] != (0+0j)
            ).map(
                __map
            ).saveAsTextFile(path)

            s.unpersist()
        elif type(state) == str:
            if os.path.exists(state):
                def __smap(m):
                    a = m.split()
                    return int(a[0]), complex(a[2])

                if self.__num_dimensions == 1:
                    def __map(m):
                        a = []
                        for p in range(num_p):
                            a.append(int(m[0] / (cs_size ** (num_p - 1 - p))) % size)
                        a.append((abs(m[1]) ** 2).real)
                        return " ".join([str(i) for i in a])
                elif self.__num_dimensions == 2:
                    def __map(m):
                        a = []
                        for p in range(num_p):
                            a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p) * size_y)) % size_x)
                            a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p))) % size_y)
                        a.append((abs(m[1]) ** 2).real)
                        return " ".join([str(i) for i in a])

                self.__spark_context.textFile(
                    state, minPartitions=self.__min_partitions
                ).map(
                    __smap
                ).filter(
                    lambda m: m[1] != (0+0j)
                ).map(
                    __map
                ).saveAsTextFile(path)
            else:
                raise FileNotFoundError
        else:
            raise Exception("Unsupported type for final state!")

        if save_mode == SAVE_MODE_MEMORY:
            full_measurement = disk_to_dense(path, dims, float)

            remove_tmp_path(path)
        else:
            full_measurement = path

        if check_probabilities(full_measurement) is False:
            raise Exception("The probabilities do not sum 1.0!")

        t2 = datetime.now()
        self.__execution_times['full_measurement'] = (t2 - t1).total_seconds()

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Full measurement in {}s\n".format(self.__execution_times['full_measurement']))

        return full_measurement

    def __filtered_measurement(self, full_measurement, save_mode):
        t1 = datetime.now()

        ndim = self.__num_dimensions
        num_p = self.__num_particles
        ind = self.__num_dimensions * self.__num_particles
        cs = coin_space(2)

        if self.__num_dimensions == 1:
            size = self.__size
            cs_size = cs.shape[0] * self.__size
            shape = (size, 1)
        elif self.__num_dimensions == 2:
            size_x = self.__size[0]
            size_y = self.__size[1]
            cs_size_x = cs.shape[0] * self.__size[0]
            cs_size_y = cs.shape[0] * self.__size[1]
            shape = (size_x, size_y)

        if sp.isspmatrix(full_measurement) or isinstance(full_measurement, np.ndarray):
            if self.__num_dimensions == 1:
                filtered_measurement = sp.dok_matrix((self.__size, 1))

                t = [0 for p in range(self.__num_particles)]

                for x in range(self.__size):
                    for p in range(self.__num_particles):
                        t[p] = x

                    if full_measurement[tuple(t)] != 0.0:
                        filtered_measurement[x, 0] = full_measurement[tuple(t)]
            elif self.__num_dimensions == 2:
                filtered_measurement = sp.dok_matrix((self.__size[0], self.__size[1]))

                t = [0 for p in range(ind)]

                for x in range(self.__size[0]):
                    for y in range(self.__size[1]):
                        for p in range(0, ind, ndim):
                            t[p] = x
                            t[p + 1] = y

                        if full_measurement[tuple(t)] != 0.0:
                            filtered_measurement[x, y] = full_measurement[tuple(t)]

            if save_mode != SAVE_MODE_MEMORY:
                filtered_measurement = sparse_to_disk(filtered_measurement.tocsc(), float, self.__min_partitions)
        elif type(full_measurement) == str:
            if self.__num_dimensions == 1:
                def __filter(m):
                    a = m.split()
                    for p in range(num_p):
                        if a[0] != a[p]:
                            return False
            elif self.__num_dimensions == 2:
                def __filter(m):
                    a = m.split()
                    for p in range(0, ind, ndim):
                        if a[0] != a[p] or a[1] != a[p + 1]:
                            return False

            path = get_tmp_path()

            self.__spark_context.textFile(
                full_measurement
            ).filter(
                __filter
            ).saveAsTextFile(path)

            if save_mode == SAVE_MODE_MEMORY:
                filtered_measurement = disk_to_dense(path, shape, float)
                remove_tmp_path(path)
            else:
                filtered_measurement = path

        t2 = datetime.now()
        self.__execution_times['filtered_measurement'] = (t2 - t1).total_seconds()

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Filtered measurement in {}s\n".format(self.__execution_times['filtered_measurement']))

        return filtered_measurement.toarray()

    def __partial_measurement(self, state, save_mode):
        t1 = datetime.now()

        num_p = self.__num_particles
        ind = self.__num_dimensions * self.__num_particles
        cs = coin_space(2)

        if self.__num_dimensions == 1:
            size = self.__size
            cs_size = cs.shape[0] * self.__size
            shape = (size, 1)
        elif self.__num_dimensions == 2:
            size_x = self.__size[0]
            size_y = self.__size[1]
            cs_size_x = cs.shape[0] * self.__size[0]
            cs_size_y = cs.shape[0] * self.__size[1]
            shape = (size_x, size_y)

        paths = []

        if sp.isspmatrix(state) or isinstance(state, np.ndarray):
            s = broadcast(self.__spark_context, state)

            if self.__num_dimensions == 1:
                for p in range(self.__num_particles):
                    def __map(m):
                        a = []
                        for p2 in range(num_p):
                            a.append(int(m / (cs_size ** (num_p - 1 - p2))) % size)
                        return "{} {} {}".format(a[p], 0, (abs(s.value[m, 0]) ** 2).real)

                    path = get_tmp_path()

                    self.__spark_context.range(
                        cs_size ** num_p
                    ).filter(
                        lambda m: s.value[m, 0] != (0+0j)
                    ).map(
                        __map
                    ).saveAsTextFile(path)

                    if check_probabilities(path, self.__spark_context) is False:
                        raise Exception("The probabilities do not sum 1.0!")

                    paths.append(path)
            elif self.__num_dimensions == 2:
                for p in range(0, ind, self.__num_dimensions):
                    def __map(m):
                        a = []
                        for p2 in range(num_p):
                            a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                            a.append(int(m / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                        return "{} {} {}".format(a[p], a[p + 1], (abs(s.value[m, 0]) ** 2).real)

                    path = get_tmp_path()

                    self.__spark_context.range(
                        (cs_size_x * cs_size_y) ** num_p
                    ).filter(
                        lambda m: s.value[m, 0] != (0+0j)
                    ).map(
                        __map
                    ).saveAsTextFile(path)

                    if check_probabilities(path, self.__spark_context) is False:
                        raise Exception("The probabilities do not sum 1.0!")

                    paths.append(path)
            s.unpersist()
        else:
            def __smap(m):
                a = m.split()
                return int(a[0]), complex(a[2])

            if self.__num_dimensions == 1:
                for p in range(self.__num_particles):
                    def __map(m):
                        a = []
                        for p2 in range(num_p):
                            a.append(int(m / (cs_size ** (num_p - 1 - p2))) % size)
                        return "{} {} {}".format(a[p], 0, (abs(m[1]) ** 2).real)

                    path = get_tmp_path()

                    self.__spark_context.textFile(
                        state, minPartitions=self.__min_partitions
                    ).map(
                        __smap
                    ).filter(
                        lambda m: m[1] != (0+0j)
                    ).map(
                        __map
                    ).saveAsTextFile(path)

                    if check_probabilities(path, self.__spark_context) is False:
                        raise Exception("The probabilities do not sum 1.0!")

                    paths.append(path)
            elif self.__num_dimensions == 2:
                for p in range(0, ind, self.__num_dimensions):
                    def __map(m):
                        a = []
                        for p2 in range(num_p):
                            a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2) * size_y)) % size_x)
                            a.append(int(m[0] / ((cs_size_x * cs_size_y) ** (num_p - 1 - p2))) % size_y)
                        return "{} {} {}".format(a[p], a[p + 1], (abs(m[1]) ** 2).real)

                    path = get_tmp_path()

                    self.__spark_context.textFile(
                        state, minPartitions=self.__min_partitions
                    ).map(
                        __smap
                    ).filter(
                        lambda m: m[1] != (0+0j)
                    ).map(
                        __map
                    ).saveAsTextFile(path)

                    if check_probabilities(path, self.__spark_context) is False:
                        raise Exception("The probabilities do not sum 1.0!")

                    paths.append(path)

        if save_mode == SAVE_MODE_MEMORY:
            partial_measurement = []

            for pm in paths:
                partial_measurement.append(disk_to_dense(pm, shape, float))
                remove_tmp_path(pm)
        else:
            partial_measurement = paths

        t2 = datetime.now()
        self.__execution_times['partial_measurement'] = (t2 - t1).total_seconds()

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Partial measurement in {}s\n".format(self.__execution_times['partial_measurement']))

        return partial_measurement

    def measure(self, state, particles=False, save_mode=SAVE_MODE_MEMORY):
        result = {}

        result['full_measurement'] = self.__full_measurement(state, save_mode)

        if self.__num_particles > 1:
            result['filtered_measurement'] = self.__filtered_measurement(result['full_measurement'], save_mode)

            if particles:
                result['partial_measurement'] = self.__partial_measurement(state, save_mode)

        self.__memory_usage['pdf'] = get_size_of(result)

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("PDFs are consuming {} bytes\n".format(self.__memory_usage['pdf']))

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
                "Coin operator in {}s\n".format(self.__execution_times['coin_operator']),
                "Shift operator tmp in {}s\n".format(self.__execution_times['shift_operator_tmp']),
                "Shift operator in {}s\n".format(self.__execution_times['shift_operator']),
                "Unitary operator tmp in {}s\n".format(self.__execution_times['unitary_operator_tmp']),
                "Unitary operator in {}s\n".format(self.__execution_times['unitary_operator']),
                "Interaction operator tmp in {}s\n".format(self.__execution_times['interaction_operator_tmp']),
                "Interaction operator in {}s\n".format(self.__execution_times['interaction_operator']),
                "Multiparticles unitary operator tmp in {}s\n".format(
                    self.__execution_times['multiparticle_unitary_operator_tmp']
                ),
                "Multiparticles unitary operator in {}s\n".format(
                    self.__execution_times['multiparticles_unitary_operator']
                ),
                "Walk operator in {}s\n".format(self.__execution_times['walk_operator']),
                "Walk in {}s\n".format(self.__execution_times['walk']),
                "Measurement tmp in {}s\n".format(self.__execution_times['measurement_tmp']),
                "Full measurement in {}s\n".format(self.__execution_times['full_measurement']),
                "Filtered measurement in {}s\n".format(self.__execution_times['filtered_measurement']),
                "Partial measurement in {}s\n".format(self.__execution_times['partial_measurement']),
                "State exportation in {}s\n".format(self.__execution_times['export_state']),
                "PDF exportation in {}s\n".format(self.__execution_times['export_pdf']),
                "PDF filtering in {}s\n".format(self.__execution_times['filter_pdf']),
                "Plots in {}s\n".format(self.__execution_times['export_plot'])
            ]

            str_times = ''.join(str_times)

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
                "Coin operator is consuming {} bytes\n".format(self.__memory_usage['coin_operator']),
                "Shift operator tmp is consuming {} bytes\n".format(self.__memory_usage['shift_operator_tmp']),
                "Shift operator is consuming {} bytes\n".format(self.__memory_usage['shift_operator']),
                "Unitary operator is consuming {} bytes\n".format(self.__memory_usage['unitary_operator_tmp']),
                "Unitary operator is consuming {} bytes\n".format(self.__memory_usage['unitary_operator']),
                "Interaction operator tmp is consuming {} bytes\n".format(
                    self.__memory_usage['interaction_operator_tmp']
                ),
                "Interaction operator is consuming {} bytes\n".format(self.__memory_usage['interaction_operator']),
                "Multiparticles unitary operator tmp is consuming {} bytes\n".format(
                    self.__memory_usage['unitary_operator']
                ),
                "Multiparticles unitary operator is consuming {} bytes\n".format(
                    self.__memory_usage['unitary_operator']
                ),
                "Walk operator is consuming {} bytes\n".format(self.__memory_usage['walk_operator']),
                "State is consuming {} bytes\n".format(self.__memory_usage['state']),
                "PDF is consuming {} bytes\n".format(self.__memory_usage['pdf'])
            ]

            str_memory = ''.join(str_memory)

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
        t1 = datetime.now()

        for k, v in pdfs.items():
            title = self.plot_title()

            if self.__num_dimensions == 1:
                if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                    axis = range(- int((self.__size - 1) / 2), int((self.__size - 1) / 2) + 1)
                else:
                    axis = range(self.__size)

                labels = "Position", "Probability"
            elif self.__num_dimensions == 2:
                if self.__walk_type == DiscreteTimeQuantumWalk.DTQW2D_WALK_LATTICE:
                    axis = np.meshgrid(
                        range(- int((self.__size[0] - 1) / 2), int((self.__size[0] - 1) / 2) + 1),
                        range(- int((self.__size[1] - 1) / 2), int((self.__size[1] - 1) / 2) + 1)
                    )
                else:
                    axis = np.meshgrid(range(self.__size[0]), range(self.__size[1]))

                labels = "Position X", "Position Y", "Probability"

            if k == 'full_measurement':
                if self.__num_dimensions == 1:
                    if self.__num_particles > 2:
                        continue

                    if self.__num_particles == 2:
                        if self.__walk_type == DiscreteTimeQuantumWalk.DTQW1D_WALK_LINE:
                            axis = np.meshgrid(
                                range(- int((self.__size - 1) / 2), int((self.__size - 1) / 2) + 1),
                                range(- int((self.__size - 1) / 2), int((self.__size - 1) / 2) + 1)
                            )
                        else:
                            axis = np.meshgrid(range(self.__size), range(self.__size))

                        labels = "Position X1", "Position X2", "Probability"

                        self.__build_twodim_plot(v, axis, labels, title)
                    else:
                        self.__build_onedim_plot(v, axis, labels, title)
                elif self.__num_dimensions == 2:
                    if self.__num_particles > 1:
                        continue

                    self.__build_twodim_plot(v, axis, labels, title)

                if path is None:
                    plt.show()
                else:
                    filename = path + self.output_filename() + "_" + k.upper() + ".png"

                    plt.savefig(filename, kwargs=kwargs)
            elif k == 'filtered_measurement':
                if self.__num_dimensions == 1:
                    self.__build_onedim_plot(v, axis, labels, title)
                elif self.__num_dimensions == 2:
                    self.__build_twodim_plot(v, axis, labels, title)

                if path is None:
                    plt.show()
                else:
                    filename = path + self.output_filename() + "_" + k.upper() + ".png"

                    plt.savefig(filename, kwargs=kwargs)
            elif k == 'partial_measurement':
                for i in range(len(v)):
                    particle_number = " (Particle " + str(i + 1) + ")"

                    if self.__num_dimensions == 1:
                        self.__build_onedim_plot(v[i], axis, labels, title + particle_number)
                    elif self.__num_dimensions == 2:
                        self.__build_twodim_plot(v[i], axis, labels, title + particle_number)

                    if path is None:
                        plt.show()
                    else:
                        filename = path + self.output_filename() + "_" + k.upper() + "_" + str(i + 1) + ".png"

                        plt.savefig(filename, kwargs=kwargs)

        t2 = datetime.now()
        self.__execution_times['export_plot'] = (t2 - t1).total_seconds()

        if DEBUG_MODE:
            with open(self.__debug_file, 'a+') as f:
                f.write("Plots in {}s\n".format(self.__execution_times['export_plot']))
