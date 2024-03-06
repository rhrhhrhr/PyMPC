import typing
from .base import *
from .ellipsoid import npl, Ellipsoid


class Polyhedron(SetBase):
    # 用线性不等式组 A @ x <= b 来表示一个多边形
    # n_edges: 边的个数
    # n_dim: 多面体维度
    # l_mat: A
    # r_vec: b

    def __init__(self, l_mat: np.ndarray, r_vec: np.ndarray):
        if l_mat.ndim != 2:
            raise SetTypeError('left matrix', 'polyhedron', '2D array')
        if r_vec.ndim != 1:
            raise SetTypeError('right vector', 'polyhedron', '1D array')
        if l_mat.shape[0] != r_vec.shape[0]:
            raise SetDimensionError('left matrix', 'right vector')

        self.__n_edges, self.__n_dim = l_mat.shape
        self.__l_mat = l_mat
        self.__r_vec = r_vec

    @property
    def n_edges(self) -> int:
        return self.__n_edges

    @property
    def n_dim(self) -> int:
        return self.__n_dim

    @property
    def l_mat(self) -> np.ndarray:
        return self.__l_mat

    @property
    def r_vec(self) -> np.ndarray:
        return self.__r_vec

    # 浅拷贝
    def __copy__(self, other: 'Polyhedron') -> None:
        self.__n_edges = other.__n_edges
        self.__n_dim = other.__n_dim
        self.__l_mat = other.__l_mat
        self.__r_vec = other.__r_vec

    # 打印该多面体的信息
    def __str__(self) -> str:
        return ('====================================================================================================\n'
                'left matrix @ x <= right vector\n'
                '====================================================================================================\n'
                f'left matrix:\n'
                f'{self.__l_mat}\n'
                '----------------------------------------------------------------------------------------------------\n'
                f'right vector:\n'
                f'{self.__r_vec}\n'
                '====================================================================================================')

    def contains(self, point: np.ndarray or cp.Expression) -> bool or cp.Constraint:
        if isinstance(point, np.ndarray):
            res = np.all(self.__l_mat @ point - self.__r_vec <= 0)
        else:
            res = self.__l_mat @ point - self.__r_vec <= 0

        return res

    # 多边形绘图会有误差，因为采用等高线来画的，增加采样点的个数可以提高画图精度
    def plot(self, ax: plt.Axes, x_lim: typing.List[int or float] = None, y_lim: typing.List[int or float] = None,
             default_bound=100, n_points=2000, color='b') -> None:
        if self.__n_dim != 2:
            raise SetPlotError()
        if x_lim is None:
            x_min = -support_fun(np.array([-1, 0]), self)
            x_max = support_fun(np.array([1, 0]), self)
            x_lim = self.get_grid_lim(x_min, x_max, default_bound)
        if y_lim is None:
            y_min = -support_fun(np.array([0, -1]), self)
            y_max = support_fun(np.array([0, 1]), self)
            y_lim = self.get_grid_lim(y_min, y_max, default_bound)

        x = np.linspace(*x_lim, n_points)
        y = np.linspace(*y_lim, n_points)
        x_grid, y_grid = np.meshgrid(x, y)

        z = np.sum(np.maximum(self.__l_mat[:, 0, np.newaxis, np.newaxis] * x_grid
                              + self.__l_mat[:, 1, np.newaxis, np.newaxis] * y_grid
                              - self.__r_vec[:, np.newaxis, np.newaxis], 0), axis=0)

        ax.contour(x_grid, y_grid, z, levels=0, colors=color)

    # 绘制一个多面体时需要知道大概范围
    @staticmethod
    def get_grid_lim(val_min: int or float, val_max: int or float, default_bound: int or float) \
            -> typing.List[int or float]:
        if val_min == -float('inf') and val_max == float('inf'):
            lim = [-default_bound, default_bound]
        elif val_min == -float('inf') and val_max != float('inf'):
            bound = abs(val_max) * 1.2
            lim = [-bound, bound]
        elif val_min != -float('inf') and val_max == float('inf'):
            bound = abs(val_min) * 1.2
            lim = [-bound, bound]
        else:
            margin = 0.1 * (val_max - val_min)
            lim = [val_min - margin, val_max + margin]

        return lim

    def subset_eq(self, other: 'Polyhedron') -> bool:
        return all([support_fun(other.__l_mat[i, :], self) <= other.__r_vec[i] for i in range(other.__n_edges)])

    # 将不等式组右侧向量归一化，防止系数过大，影响支撑函数（线性规划）求解
    def normalization(self) -> None:
        r_vec_abs = np.where(self.__r_vec == 0, 1, np.abs(self.__r_vec))
        self.__l_mat = self.__l_mat / r_vec_abs[:, np.newaxis]
        self.__r_vec = self.__r_vec / r_vec_abs

    # 去除某个边
    def remove_edge(self, edge: int or list) -> 'Polyhedron':
        l_mat = np.delete(self.__l_mat, edge, 0)
        r_vec = np.delete(self.__r_vec, edge, 0)

        return self.__class__(l_mat, r_vec)

    # 去除冗余项
    def remove_redundant_term(self) -> None:
        # 一条边不可能冗余
        if self.__n_edges >= 2:
            i = 0
            while i < self.__n_edges:
                without_row_i = self.remove_edge(i)
                s_i = self.__r_vec[i] - support_fun(self.__l_mat[i, :], without_row_i)

                if s_i >= 0:
                    self.__copy__(without_row_i)
                else:
                    i = i + 1

    # 傅里叶-莫茨金消元法，这里从最后一个元素开始倒着消除，因此使用该方法前应该把需要保留的元素放在最前面
    # 相当于给一个变量左乘矩阵
    # [1 0 0 ... 0 0 ... 0]
    # [0 1 0 ... 0 0 ... 0]
    # [. . . ... . . ... .]
    # [0 0 0 ... 1 0 ... 0]
    #              | --- |
    #                 |
    #                 V
    #              减少的维度
    def fourier_motzkin_elimination(self, n_dim: int) -> None:
        if n_dim < 0:
            raise SetTypeError('eliminated dimension', 'polyhedron', 'positive integer')
        for _ in range(n_dim):
            pos_a = np.empty((0, self.__n_dim - 1))
            pos_b = np.empty(0)
            neg_a = np.empty((0, self.__n_dim - 1))
            neg_b = np.empty(0)

            new_a = np.empty((0, self.__n_dim - 1))
            new_b = np.empty(0)

            for i in range(self.__n_edges):
                a_i_last = self.__l_mat[i, -1]
                a_i_others = self.__l_mat[i, :-1]
                b_i = self.__r_vec[i]

                if a_i_last > 0:
                    pos_a = np.vstack((pos_a, a_i_others / a_i_last))
                    pos_b = np.hstack((pos_b, b_i / a_i_last))
                elif a_i_last == 0:
                    new_a = np.vstack((new_a, a_i_others))
                    new_b = np.hstack((new_b, b_i))
                else:
                    neg_a = np.vstack((neg_a, -a_i_others / a_i_last))
                    neg_b = np.hstack((neg_b, -b_i / a_i_last))

            for i in range(len(pos_a)):
                for j in range(len(neg_a)):
                    new_a = np.vstack((new_a, pos_a[i, :] + neg_a[j, :]))
                    new_b = np.hstack((new_b, pos_b[i] + neg_b[j]))

            self.__init__(new_a, new_b)
            self.remove_redundant_term()

    # 相当于给一个变量左乘矩阵
    # [1 0 0 ... 0]
    # [0 1 0 ... 0]
    # [. . . ... .]
    # [0 0 0 ... 1]
    # [0 0 0 ... 0] ---
    # [. . . ... .]   | ---> 增加的维度
    # [0 0 0 ... 0] ---
    def extend_dimensions(self, n_dim: int) -> None:
        if n_dim < 0:
            raise SetTypeError('extended dimension', 'polyhedron', 'positive integer')
        elif n_dim > 0:
            zero_1 = np.zeros((self.__n_edges, n_dim))
            zero_2 = np.zeros((n_dim, self.__n_dim))
            zero_3 = np.zeros(2 * n_dim)
            eye = np.eye(n_dim)
            self.__n_edges = self.__n_edges + 2 * n_dim
            self.__n_dim = self.__n_dim + n_dim
            self.__l_mat = np.block([[self.__l_mat, zero_1], [zero_2, eye], [zero_2, -eye]])
            self.__r_vec = np.block([self.__r_vec, zero_3])

    # 闵可夫斯基和（或平移）
    def __add__(self, other: 'Polyhedron' or np.ndarray) -> 'Polyhedron':
        if isinstance(other, Polyhedron):
            h_self_other = np.array([support_fun(self.__l_mat[i, :], other) for i in range(self.__n_edges)])
            h_other_self = np.array([support_fun(other.__l_mat[i, :], self) for i in range(other.__n_edges)])

            res_l_mat = np.vstack((self.__l_mat, other.__l_mat))
            res_r_vec = np.hstack((self.__r_vec + h_self_other, other.__r_vec + h_other_self))

            res = self.__class__(res_l_mat, res_r_vec)
            res.remove_redundant_term()

        else:
            if other.ndim != 1:
                raise SetCalculationError('polyhedron', 'added', '1D array')
            if other.size != self.__n_dim:
                raise SetCalculationError('polyhedron', 'added', 'array with matching dimension')

            res = self.__class__(self.__l_mat, self.__r_vec + self.__l_mat @ other)

        res.normalization()

        return res

    def __neg__(self) -> 'Polyhedron':
        return self.__class__(-self.__l_mat, self.__r_vec)

    # 特别的，这里指庞特里亚金差，即闵可夫斯基和的逆运算
    # 即若 p2 = p1 + p3，则 p3 = p2 - p1，只有当输入为一个点（数组）时该运算等价于 (-p1) + p2
    def __sub__(self, other: 'Polyhedron' or np.ndarray) -> 'Polyhedron':
        if isinstance(other, Polyhedron):
            h_self_other = np.array([support_fun(self.__l_mat[i, :], other) for i in range(self.__n_edges)])

            res = self.__class__(self.__l_mat, self.__r_vec - h_self_other)
            res.remove_redundant_term()
            res.normalization()

        else:
            res = self + (-other)

        return res

    # 多面体坐标变换，Poly_new = Poly @ mat 意味着 Poly 是将 Poly_new 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    def __matmul__(self, other: np.ndarray) -> 'Polyhedron':
        if other.ndim != 2:
            raise SetCalculationError('polyhedron', 'multiplied', '2D array')
        if other.shape[0] != self.__n_dim:
            raise SetCalculationError('polyhedron', 'multiplied', 'array with matching dimension')

        return self.__class__(self.__l_mat @ other, self.__r_vec)

    # 多面体坐标变换，Poly_new = Poly @ mat 意味着 Poly_new 是将 Poly 中的所有点通过 mat 映射后的区域，这一定义是为了方便计算不变集
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> 'Polyhedron' or NotImplemented:
        if ufunc == np.matmul:
            lhs, rhs = inputs
            row, col = lhs.shape
            u, s, v = npl.svd(lhs)
            rank = s.shape[0]

            res = self.__matmul__(v.T)
            res.fourier_motzkin_elimination(col - rank)
            res = res.__matmul__(np.diag(1 / s))
            res.extend_dimensions(row - rank)
            res = res.__matmul__(u.T)
        elif ufunc == np.add:
            lhs, rhs = inputs
            res = self.__add__(lhs)
        else:
            res = NotImplemented

        return res

    def __mul__(self, other: int or float) -> 'Polyhedron':
        if other < 0:
            raise SetCalculationError('polyhedron', 'multiplied', 'positive number')

        return self.__class__(self.__l_mat / other, self.__r_vec)

    def __and__(self, other: 'Polyhedron') -> 'Polyhedron':
        res = self.__class__(np.vstack((self.__l_mat, other.__l_mat)), np.hstack((self.__r_vec, other.__r_vec)))
        res.remove_redundant_term()

        return res

    def __eq__(self, other: 'Polyhedron') -> bool:
        return self.subset_eq(other) and other.subset_eq(self)

    def get_max_ellipsoid(self, p: np.ndarray, center: np.ndarray = None) -> Ellipsoid:
        ellipsoid_center = np.zeros(self.__n_dim) if center is None else center

        if not self.contains(ellipsoid_center):
            raise SetError('Cannot find a maximum ellipsoid since the center is not in the polyhedron!')

        p_bar = npl.cholesky(p)
        r_vec_bar = self.__r_vec + self.__l_mat @ ellipsoid_center
        l_mat_bar = self.__l_mat @ npl.inv(p_bar).T
        min_center_to_edge_distance = np.min(np.abs(r_vec_bar) / npl.norm(l_mat_bar, ord=2, axis=1))

        return Ellipsoid(p, min_center_to_edge_distance ** 2, center)


class Rn(Polyhedron):
    def __init__(self, dim: int):
        if dim <= 0:
            raise SetTypeError('dimension', 'Rn', 'positive integer')

        super().__init__(np.zeros((1, dim)), np.zeros(1))

    def __str__(self):
        return f'Vector space: dimension -> {self.n_dim}.'


class UnitCube(Polyhedron):
    def __init__(self, dim: int, side_length: int or float):
        if dim <= 0:
            raise SetTypeError('dimension', 'unit cube', 'positive integer')
        if side_length < 0:
            raise SetTypeError('side length', 'unit cube', 'non-negative real number')

        self.__side_length = side_length
        eye = np.eye(dim)

        super().__init__(np.vstack((eye, -eye)), (side_length / 2) * np.ones(2 * dim))

    @property
    def side_length(self) -> int or float:
        return self.__side_length

    def __str__(self) -> str:
        return f'Unit cube: dimension -> {self.n_dim}, side length -> {self.__side_length}.'
