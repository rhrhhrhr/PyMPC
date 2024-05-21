import abc
import copy
import cvxpy as cp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import typing
from .. import set


class MPCTypeException(Exception):
    def __init__(self, name: str, tp: str):
        message = 'The type of ' + name + ' must be ' + tp + '!'
        super().__init__(message)


class MPCDimensionException(Exception):
    def __init__(self, *obj: str):
        message = 'The dimensions of ' + ', '.join(obj) + ' do not match!'
        super().__init__(message)


class MPCTerminalSetTypeException(Exception):
    def __init__(self):
        super().__init__('The terminal set type must be \'zero\', \'ellipsoid\' or \'polyhedron\'')


class MPCNotImplementedException(Exception):
    def __init__(self, function: str):
        message = 'The function ' + function + ' has not been implemented yet!'
        super().__init__(message)


class LQR:
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray):
        if not (a.ndim == b.ndim == q.ndim == r.ndim == 2):
            raise MPCTypeException('A, B, Q, R', '2D array')
        if not (a.shape[0] == a.shape[1] == b.shape[0] == q.shape[0] == q.shape[1]):
            raise MPCDimensionException('A, B, Q')
        if not (b.shape[1] == r.shape[0] == r.shape[1]):
            raise MPCDimensionException('B, R')

        self.__state_dim = a.shape[1]
        self.__input_dim = b.shape[1]

        self.__a = a
        self.__b = b
        self.__q = q
        self.__r = r

        self.__p, self.__k = self.cal_lqr()

    @property
    def state_dim(self) -> int:
        return self.__state_dim

    @property
    def input_dim(self) -> int:
        return self.__input_dim

    @property
    def a(self) -> np.ndarray:
        return self.__a

    @property
    def b(self) -> np.ndarray:
        return self.__b

    @property
    def q(self) -> np.ndarray:
        return self.__q

    @property
    def r(self) -> np.ndarray:
        return self.__r

    @property
    def p(self) -> np.ndarray:
        return self.__p

    @property
    def k(self) -> np.ndarray:
        return self.__k

    def cal_lqr(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        p = spl.solve_discrete_are(self.__a, self.__b, self.__q, self.__r)
        k = npl.inv(self.__r + self.__b.T @ p @ self.__b) @ self.__b.T @ p @ self.__a

        return p, k

    def __call__(self, real_time_state: np.ndarray) -> np.ndarray:
        if real_time_state.ndim != 1:
            raise MPCTypeException('real time state', '1D array')
        if real_time_state.shape[0] != self.__state_dim:
            raise MPCDimensionException('real time state and state in controller')

        return -self.__k @ real_time_state


class MPCBase(LQR, metaclass=abc.ABCMeta):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 terminal_set_type: str, solver: str):
        super().__init__(a, b, q, r)

        if pred_horizon <= 0:
            raise MPCTypeException('prediction horizon', 'positive integer')
        if terminal_set_type not in ['zero', 'ellipsoid', 'polyhedron']:
            raise MPCTerminalSetTypeException()

        self.__pred_horizon = pred_horizon

        self.__real_time_state = cp.Parameter(self.state_dim)
        self.__state_series = cp.Variable(self.state_dim * (pred_horizon + 1))
        self.__input_series = cp.Variable(self.input_dim * pred_horizon)

        self.__terminal_set_type = terminal_set_type

        self.__solver = solver

    @property
    def pred_horizon(self) -> int:
        return self.__pred_horizon

    @pred_horizon.setter
    def pred_horizon(self, value: int) -> None:
        if value <= 0:
            raise MPCTypeException('prediction horizon', 'positive integer')

        self.__pred_horizon = value

    @property
    @abc.abstractmethod
    def state_set(self) -> set.Polyhedron:
        ...

    @property
    @abc.abstractmethod
    def input_set(self) -> set.Polyhedron:
        ...

    @property
    @abc.abstractmethod
    def terminal_set(self) -> set.Polyhedron or set.Ellipsoid:
        ...

    @property
    def state_prediction_series(self) -> np.ndarray:
        return self.__state_series.value.reshape(self.__pred_horizon + 1, self.state_dim).T

    @property
    def input_prediction_series(self) -> np.ndarray:
        return self.__input_series.value.reshape(self.__pred_horizon, self.input_dim).T

    @property
    def input_ini(self) -> cp.Variable:
        return self.__input_series[0:self.input_dim]

    @property
    def state_ini(self) -> cp.Variable:
        return self.__state_series[0:self.state_dim]

    @property
    def real_time_state(self) -> cp.Parameter:
        return self.__real_time_state

    @real_time_state.setter
    def real_time_state(self, value: np.ndarray) -> None:
        if value.ndim != 1:
            raise MPCTypeException('real time state', '1D array')
        if value.size != self.state_dim:
            raise MPCDimensionException('real time state and state in controller')

        self.__real_time_state.value = value

    @property
    def terminal_set_type(self) -> str:
        return self.__terminal_set_type

    @terminal_set_type.setter
    def terminal_set_type(self, value: str) -> None:
        self.__terminal_set_type = value

    @property
    def solver(self) -> str:
        return self.__solver

    @solver.setter
    def solver(self, value) -> None:
        self.__solver = value

    @property
    @abc.abstractmethod
    def initial_constraint(self) -> cp.Constraint:
        ...

    @property
    @abc.abstractmethod
    def problem(self):
        ...

    @abc.abstractmethod
    def __call__(self, real_time_state: np.ndarray) -> np.ndarray:
        ...

    def cal_terminal_set(self) -> set.Polyhedron:
        # 在终端约束 Xf 内的一点 x 满足：
        # 1. 当采用控制律 u = Kx 时，状态约束和输入约束均满足 -- 这一条件描述的集合为 X 与 U @ K 的交集，集合与矩阵的乘法解释请参考文件poly
        # 2. 下一时刻的状态 x+ = A_k @ x 仍属于 Xf -- 这一条件描述的集合 set 被包含于 set @ A_k
        # 若设置终端约束集合为原点，则生成一个边长为0的单位立方体，否则计算最大的满足上述条件的集合
        if self.__terminal_set_type == 'zero':
            terminal_set = set.unit_cube(self.state_dim, 0)
        elif self.__terminal_set_type == 'ellipsoid':
            state_set_in_terminal = self.state_set & (self.input_set @ self.k)
            terminal_set = state_set_in_terminal.get_max_ellipsoid(self.p / 2)
        else:
            set_k = self.state_set & (self.input_set @ self.k)
            terminal_set = copy.deepcopy(set_k)
            a_k = self.a - self.b @ self.k

            while True:
                set_k = set_k @ a_k
                terminal_set_next = terminal_set & set_k

                if terminal_set.subset_eq(terminal_set_next):
                    break

                terminal_set = terminal_set_next

        return terminal_set

    def construct_problem(self) -> cp.Problem:
        cost = 0
        state_k = self.__state_series[0:self.state_dim]

        # 对于初始状态的约束
        constraints = [self.initial_constraint]

        for k in range(self.__pred_horizon):
            input_k = self.__input_series[k * self.input_dim:(k + 1) * self.input_dim]

            # l(x, u) = x.T @ Q @ x + u.T @ R @ u
            cost = cost + (state_k @ self.q @ state_k + input_k @ self.r @ input_k) / 2

            # x^+ = A @ x + B @ u
            state_k_next = self.__state_series[(k + 1) * self.state_dim:(k + 2) * self.state_dim]
            constraints.append(state_k_next == self.a @ state_k + self.b @ input_k)

            # x in X, u in U
            constraints.append(self.state_set.contains(state_k))
            constraints.append(self.input_set.contains(input_k))

            state_k = state_k_next

        # 另一种构建优化问题的思路，只把输入序列当作决策变量 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        #
        # constraints = []
        # state_k = self.__real_time_state
        #
        # for k in range(self.__pred_horizon):
        #     input_k = self.__input_series[k * self.input_dim:(k + 1) * self.input_dim]
        #
        #     cost = cost + (state_k @ self.q @ state_k + input_k @ self.r @ input_k)
        #     constraints.append(state_set.is_interior_point(state_k) <= 0)
        #     constraints.append(input_set.is_interior_point(input_k) <= 0)
        #
        #     state_k = self.a @ state_k + self.b @ input_k
        #
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = =

        cost = cost + (state_k @ self.p @ state_k) / 2
        if self.__terminal_set_type == 'zero':
            constraints.append(state_k == 0)
        else:
            constraints.append(self.terminal_set.contains(state_k))

        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem

    def cal_feasible_set(self) -> set.Polyhedron:
        # 这里先求出了M，C矩阵，于是知道了Xk = M*x + C*Uk，之后将约束条件转化为G*Xk <= h，再包含A_Uk*Uk <= b_Uk
        # 于是有
        # [G*M G*C ][x ]      [h   ]
        # [        ][  ]  <=  [    ]
        # [ 0  A_Uk][Uk]      [b_Uk]
        # a_bar, b_bar分别代表上面两个矩阵

        if self.__terminal_set_type == 'ellipsoid':
            raise MPCNotImplementedException('calculation for the feasible set when the terminal set is a ellipsoid')
        # 生成矩阵 M
        m = np.zeros((self.state_dim * (self.__pred_horizon + 1), self.state_dim))
        a_i = np.eye(self.state_dim)

        for i in range(self.pred_horizon + 1):
            m[i * self.state_dim:(i + 1) * self.state_dim, :] = a_i
            a_i = a_i @ self.a

        # 生成矩阵 C
        c = spl.block_diag(*[self.b for _ in range(self.__pred_horizon)])

        for i in range(self.__pred_horizon - 1):
            c[(i + 1) * self.state_dim:(i + 2) * self.state_dim, :] \
                += self.a @ c[i * self.state_dim:(i + 1) * self.state_dim, :]

        zero = np.zeros((self.state_dim, self.input_dim * self.__pred_horizon))
        c = np.vstack((zero, c))

        # 生成 G 和 h
        g = spl.block_diag(*[self.state_set.l_mat for _ in range(self.__pred_horizon)], self.terminal_set.l_mat)
        h = np.hstack((np.tile(self.state_set.r_vec, self.pred_horizon), self.terminal_set.r_vec))

        # 生成 A_Uk 和 b_Uk
        a_uk = spl.block_diag(*[self.input_set.l_mat for _ in range(self.__pred_horizon)])
        b_uk = np.tile(self.input_set.r_vec, self.__pred_horizon)

        # 生成对应的零矩阵部分
        zero = np.zeros((a_uk.shape[0], self.state_dim))

        l_mat = np.block([[g @ m, g @ c], [zero, a_uk]])
        r_vec = np.hstack((h, b_uk))

        feasible_set = set.Polyhedron(l_mat, r_vec)

        # 傅里叶-莫茨金消元法，将控制输入变量U消去
        feasible_set.fourier_motzkin_elimination(self.input_dim * self.__pred_horizon)

        return feasible_set
