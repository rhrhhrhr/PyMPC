import abc
import copy
import cvxpy as cp
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
from functools import cached_property
from numpy.typing import NDArray
from ..set import Polyhedron, Ellipsoid, unit_cube


def solve_discrete_lqr(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    q: NDArray[np.float64],
    r: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p = spl.solve_discrete_are(a, b, q, r)
    k = npl.inv(r + b.T @ p @ b) @ b.T @ p @ a

    return p, k


class LQR:
    def __init__(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        q: NDArray[np.float64],
        r: NDArray[np.float64],
    ):
        _, self._k = solve_discrete_lqr(a, b, q, r)

    def __call__(self, real_time_state: NDArray[np.float64]) -> NDArray[np.float64]:
        return -self._k @ real_time_state


class MPCBase(metaclass=abc.ABCMeta):
    def __init__(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        q: NDArray[np.float64],
        r: NDArray[np.float64],
        pred_horizon: int,
        terminal_set_type: str = "polyhedron",
        solver: str = "OSQP",
    ):
        self._a = a
        self._b = b
        self._q = q
        self._r = r

        self._p, self._k = solve_discrete_lqr(a, b, q, r)

        self._state_dim = a.shape[1]
        self._input_dim = b.shape[1]

        self._pred_horizon = pred_horizon

        self._real_time_state = cp.Parameter(self._state_dim)
        self._state_series = cp.Variable(self._state_dim * (pred_horizon + 1))
        self._input_series = cp.Variable(self._input_dim * pred_horizon)

        self._terminal_set_type = terminal_set_type
        self._terminal_set = self._compute_terminal_set(terminal_set_type)

        self._solver = solver

    @property
    @abc.abstractmethod
    def state_set(self) -> Polyhedron: ...

    @property
    @abc.abstractmethod
    def input_set(self) -> Polyhedron: ...

    @property
    def state_series(self) -> NDArray[np.float64]:
        state_series = self._state_series.value
        if state_series is None:
            raise ValueError("The optimization problem has not been solved yet.")

        return state_series.reshape(self._pred_horizon + 1, self._state_dim).T

    @property
    def input_series(self) -> NDArray[np.float64]:
        input_series = self._input_series.value
        if input_series is None:
            raise ValueError("The optimization problem has not been solved yet.")

        return input_series.reshape(self._pred_horizon, self._input_dim).T

    @property
    def real_time_state(self) -> cp.Parameter:
        return self._real_time_state

    @property
    def solver(self) -> str:
        return self._solver

    @solver.setter
    def solver(self, value) -> None:
        self._solver = value

    @property
    @abc.abstractmethod
    def initial_constraint(self) -> cp.Constraint: ...

    @abc.abstractmethod
    def __call__(self, real_time_state: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def _compute_terminal_set(self, key: str) -> Polyhedron | Ellipsoid:
        # 在终端约束 Xf 内的一点 x 满足：
        # 1. 当采用控制律 u = Kx 时，状态约束和输入约束均满足 -- 这一条件描述的集合为 X 与 U @ K 的交集，集合与矩阵的乘法解释请参考文件poly
        # 2. 下一时刻的状态 x+ = A_k @ x 仍属于 Xf -- 这一条件描述的集合 set 被包含于 set @ A_k
        # 若设置终端约束集合为原点，则生成一个边长为0的单位立方体，否则计算最大的满足上述条件的集合
        if key == "zero":
            return unit_cube(self._state_dim, 0)
        elif key == "ellipsoid":
            state_set_in_terminal = self.state_set & (self.input_set.map_inv(self._k))
            return state_set_in_terminal.get_max_ellipsoid(self._p / 2)
        elif key == "polyhedron":
            set_k = self.state_set & (self.input_set.map_inv(self._k))
            terminal_set = copy.deepcopy(set_k)
            a_k = self._a - self._b @ self._k

            while True:
                set_k = set_k.map_inv(a_k)
                terminal_set_next = terminal_set & set_k

                if terminal_set.subset_eq(terminal_set_next):
                    break

                terminal_set = terminal_set_next

            return terminal_set
        else:
            raise ValueError(f"Unsupported terminal set type: {key}.")

    @cached_property
    def problem(self) -> cp.Problem:
        cost = 0
        state_k = self._state_series[0 : self._state_dim]

        # 对于初始状态的约束
        constraints = [self.initial_constraint]

        for k in range(self._pred_horizon):
            input_k = self._input_series[
                k * self._input_dim : (k + 1) * self._input_dim
            ]

            # l(x, u) = x.T @ Q @ x + u.T @ R @ u
            cost = (
                cost + (state_k @ self._q @ state_k + input_k @ self._r @ input_k) / 2
            )

            # x^+ = A @ x + B @ u
            state_k_next = self._state_series[
                (k + 1) * self._state_dim : (k + 2) * self._state_dim
            ]
            constraints.append(state_k_next == self._a @ state_k + self._b @ input_k)

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

        cost = cost + (state_k @ self._p @ state_k) / 2
        if self._terminal_set_type == "zero":
            constraints.append(state_k == 0)
        else:
            constraints.append(self._terminal_set.contains(state_k))

        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem

    @cached_property
    def feasible_set(self) -> Polyhedron:
        # 这里先求出了M，C矩阵，于是知道了Xk = M*x + C*Uk，之后将约束条件转化为G*Xk <= h，再包含A_Uk*Uk <= b_Uk
        # 于是有
        # [G*M G*C ][x ]      [h   ]
        # [        ][  ]  <=  [    ]
        # [ 0  A_Uk][Uk]      [b_Uk]
        # a_bar, b_bar分别代表上面两个矩阵

        if isinstance(self._terminal_set, Ellipsoid):
            raise NotImplementedError(
                "Feasible set computation is not implemented for ellipsoidal terminal sets."
            )

        # 生成矩阵 M
        m = np.zeros((self._state_dim * (self._pred_horizon + 1), self._state_dim))
        a_i = np.eye(self._state_dim)

        for i in range(self._pred_horizon + 1):
            m[i * self._state_dim : (i + 1) * self._state_dim, :] = a_i
            a_i = a_i @ self._a

        # 生成矩阵 C
        c = spl.block_diag(*[self._b for _ in range(self._pred_horizon)])

        for i in range(self._pred_horizon - 1):
            c[(i + 1) * self._state_dim : (i + 2) * self._state_dim, :] += (
                self._a @ c[i * self._state_dim : (i + 1) * self._state_dim, :]
            )

        zero = np.zeros((self._state_dim, self._input_dim * self._pred_horizon))
        c = np.vstack((zero, c))

        # 生成 G 和 h
        g = spl.block_diag(
            *[self.state_set.l_mat for _ in range(self._pred_horizon)],
            self._terminal_set.l_mat,
        )
        h = np.hstack(
            (
                np.tile(self.state_set.r_vec, self._pred_horizon),
                self._terminal_set.r_vec,
            )
        )

        # 生成 A_Uk 和 b_Uk
        a_uk = spl.block_diag(
            *[self.input_set.l_mat for _ in range(self._pred_horizon)]
        )
        b_uk = np.tile(self.input_set.r_vec, self._pred_horizon)

        # 生成对应的零矩阵部分
        zero = np.zeros((a_uk.shape[0], self._state_dim))

        l_mat = np.block([[g @ m, g @ c], [zero, a_uk]])
        r_vec = np.hstack((h, b_uk))

        feasible_set = Polyhedron(l_mat, r_vec)

        # 傅里叶-莫茨金消元法，将控制输入变量U消去
        feasible_set.reduce_dimension(self._input_dim * self._pred_horizon)

        return feasible_set
