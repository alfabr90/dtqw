import scipy.sparse as sp

from .operator import Operator
from .utils import get_tmp_path, braket, broadcast, remove_tmp_path

__all__ = ['Mesh', 'is_mesh',
           'MESH_1D_LINE', 'MESH_1D_SEGMENT', 'MESH_1D_CYCLE',
           'MESH_2D_LATTICE_DIAGONAL', 'MESH_2D_LATTICE_NATURAL',
           'MESH_2D_BOX_DIAGONAL', 'MESH_2D_BOX_NATURAL',
           'MESH_2D_TORUS_DIAGONAL', 'MESH_2D_TORUS_NATURAL']


MESH_1D_LINE = 0
MESH_1D_CYCLE = 1
MESH_1D_SEGMENT = 2

MESH_2D_LATTICE_DIAGONAL = 2 ** 10
MESH_2D_LATTICE_NATURAL = 2 ** 10 + 1
MESH_2D_TORUS_DIAGONAL = 2 ** 11
MESH_2D_TORUS_NATURAL = 2 ** 11 + 1
MESH_2D_BOX_DIAGONAL = 2 ** 12
MESH_2D_BOX_NATURAL = 2 ** 12 + 1


class Mesh:
    def __init__(self, type, size):
        self.__type = type
        self.__size = self.__define_size(size)

    @property
    def type(self):
        return self.__type

    @property
    def size(self):
        return self.__size

    def __validate(self, size):
        if self.is_1d():
            if isinstance(size, (list, tuple)):
                return False
            elif size <= 0:
                return False
        elif self.is_2d():
            if isinstance(size, (list, tuple)):
                if len(size) != 2:
                    return False
            else:
                return False
        else:
            raise NotImplementedError

        return True

    def __define_size(self, size):
        if not self.__validate(size):
            raise ValueError("not a valid size")

        if self.__type == MESH_1D_LINE:
            return 2 * size + 1
        elif self.__type == MESH_2D_LATTICE_DIAGONAL or self.__type == MESH_2D_LATTICE_NATURAL:
            return 2 * size[0] + 1, 2 * size[0] + 1
        else:
            return size

    def is_1d(self):
        return self.__type == MESH_1D_LINE or self.__type == MESH_1D_SEGMENT or self.__type == MESH_1D_CYCLE

    def is_2d(self):
        return self.__type == MESH_2D_LATTICE_DIAGONAL or self.__type == MESH_2D_LATTICE_NATURAL \
            or self.__type == MESH_2D_BOX_DIAGONAL or self.__type == MESH_2D_BOX_NATURAL \
            or self.__type == MESH_2D_TORUS_DIAGONAL or self.__type == MESH_2D_TORUS_NATURAL

    def create_operator(self, spark_context, min_partitions=8, log_filename='log.txt'):
        path = get_tmp_path()

        cs = 2
        buffer_size = 8196

        if self.__type == MESH_1D_LINE:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for x in range(self.__size):
                        f.write("{} {} {} {}\n".format(i, i, (x + l) % self.__size, x))
        elif self.__type == MESH_1D_SEGMENT:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for x in range(self.__size):
                        if x + l >= self.__size or x + l < 0:
                            bl = 0
                        else:
                            bl = l
                        f.write("{} {} {} {}\n".format(i + bl, 1 - i, x + bl, x))
        elif self.__type == MESH_1D_CYCLE:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for x in range(self.__size):
                        f.write("{} {} {} {}\n".format(i, i, (x + l) % self.__size, x))
        elif self.__type == MESH_2D_LATTICE_DIAGONAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l1 = (-1) ** i
                    for j in range(cs):
                        l2 = (-1) ** j
                        for x in range(self.__size[0]):
                            for y in range(self.__size[1]):
                                f.write(
                                    "{} {} {} {} {} {} {} {}\n".format(
                                        i, j, i, j,
                                        (x + l1) % self.__size[0],
                                        (y + l2) % self.__size[1],
                                        x, y
                                    )
                                )
        elif self.__type == MESH_2D_LATTICE_NATURAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for j in range(cs):
                        delta = int(braket(sp.identity(cs, format='csc')[:, i], sp.identity(cs, format='csc')[:, j]))
                        for x in range(self.__size[0]):
                            for y in range(self.__size[1]):
                                f.write(
                                    "{} {} {} {} {} {} {} {}\n".format(
                                        i, j, i, j,
                                        (x + l * (1 - delta)) % self.__size[0],
                                        (y + l * delta) % self.__size[1],
                                        x, y
                                    )
                                )
        elif self.__type == MESH_2D_BOX_DIAGONAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l1 = (-1) ** i
                    for j in range(cs):
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
        elif self.__type == MESH_2D_BOX_NATURAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for j in range(cs):
                        delta = int(braket(sp.identity(cs, format='csc')[:, i], sp.identity(cs, format='csc')[:, j]))
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
                                        i + bl, abs(j + bl) % cs, 1 - i, 1 - j,
                                        x + bl * (1 - delta), y + bl * delta, x, y
                                    )
                                )
        elif self.__type == MESH_2D_TORUS_DIAGONAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l1 = (-1) ** i
                    for j in range(cs):
                        l2 = (-1) ** j
                        for x in range(self.__size[0]):
                            for y in range(self.__size[1]):
                                f.write(
                                    "{} {} {} {} {} {} {} {}\n".format(
                                        i, j, i, j,
                                        (x + l1) % self.__size[0],
                                        (y + l2) % self.__size[1],
                                        x, y
                                    )
                                )
        elif self.__type == MESH_2D_TORUS_NATURAL:
            with open(path, 'w') as f:
                for i in range(cs):
                    l = (-1) ** i
                    for j in range(cs):
                        delta = int(braket(sp.identity(cs, format='csc')[:, i], sp.identity(cs, format='csc')[:, j]))
                        for x in range(self.__size[0]):
                            for y in range(self.__size[1]):
                                f.write(
                                    "{} {} {} {} {} {} {} {}\n".format(
                                        i, j, i, j,
                                        (x + l * (1 - delta)) % self.__size[0],
                                        (y + l * delta) % self.__size[1],
                                        x, y
                                    )
                                )
        else:
            # TODO
            raise NotImplementedError

        c = broadcast(spark_context, sp.identity(cs, format='csc'))

        if self.is_1d():
            shape = (cs * self.__size, cs * self.__size)
            s_csc = broadcast(spark_context, sp.identity(self.__size).tocsc())
            s_csr = broadcast(spark_context, sp.identity(self.__size).tocsr())

            def __map(m):
                a = [int(i) for i in m.split()]
                return sp.kron(
                    c.value[:, a[0]] * c.value[a[1], :],
                    s_csc.value[:, a[2]] * s_csr.value[a[3], :],
                    'csc'
                )
        elif self.is_2d():
            shape = (cs * cs * self.__size[0] * self.__size[1], cs * cs * self.__size[0] * self.__size[1])
            s_csc = broadcast(spark_context, (sp.identity(self.__size[0]).tocsc(), sp.identity(self.__size[1]).tocsc()))
            s_csr = broadcast(spark_context, (sp.identity(self.__size[0]).tocsr(), sp.identity(self.__size[1]).tocsr()))

            def __map(m):
                a = [int(i) for i in m.split()]
                return sp.kron(
                    sp.kron(c.value[:, a[0]], c.value[:, a[1]]) *
                    sp.kron(c.value[a[2], :], c.value[a[3], :]),
                    sp.kron(s_csc.value[0][:, a[4]], s_csc.value[1][:, a[5]]) *
                    sp.kron(s_csr.value[0][a[6], :], s_csr.value[1][a[7], :]),
                    'csc'
                )
        else:
            # TODO
            raise NotImplementedError

        sparse = spark_context.textFile(
            path, minPartitions=min_partitions
        ).map(
            __map
        ).reduce(
            lambda a, b: a + b
        )

        c.unpersist()
        s_csc.unpersist()
        s_csr.unpersist()

        remove_tmp_path(path)

        return Operator(sparse, spark_context, log_filename=log_filename)


def is_mesh(obj):
    return isinstance(obj, Mesh)
