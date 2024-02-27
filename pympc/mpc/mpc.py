import copy
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spl
import cvxpy as cp
from .. import poly


class MPCException(Exception):
    pass


class LQR(object):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray):
        if not (a.ndim == b.ndim == q.ndim == r.ndim == 2):
            raise MPCException('The parameters a,b,q,r must be 2D array!')
        if not (a.shape[0] == a.shape[1] == b.shape[0] == q.shape[0] == q.shape[1]):
            raise MPCException('The dimensions of the state parameters do not match!')
        if not (b.shape[1] == r.shape[0] == r.shape[1]):
            raise MPCException('The dimensions of the input parameters do not match!')

        self.__state_dim = a.shape[1]
        self.__input_dim = b.shape[1]

        self.__a = a
        self.__b = b
        self.__q = q
        self.__r = r

        self.__p, self.__k = self.cal_lqr()

    @property
    def state_dim(self):
        return self.__state_dim

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def a(self):
        return self.__a

    @property
    def b(self):
        return self.__b

    @property
    def q(self):
        return self.__q

    @property
    def r(self):
        return self.__r

    @property
    def p(self):
        return self.__p

    @property
    def k(self):
        return self.__k

    def cal_lqr(self):
        p = spl.solve_discrete_are(self.__a, self.__b, self.__q, self.__r)
        k = npl.inv(self.__r + self.__b.T @ p @ self.__b) @ self.__b.T @ p @ self.__a

        return p, k

    def __call__(self, real_time_state: np.ndarray):
        if real_time_state.ndim != 1:
            raise MPCException('The state must be a 1D array!')
        if real_time_state.shape[0] != self.__state_dim:
            raise MPCException('The dimension of the state is wrong!')

        return -self.__k @ real_time_state


class MPCBase(LQR):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 zero_terminal_set, solver):
        super().__init__(a, b, q, r)

        if pred_horizon <= 0:
            raise MPCException('The prediction horizon must be a positive integer!')

        self.__pred_horizon = pred_horizon

        self.__real_time_state = cp.Parameter(self.state_dim)
        self.__state_series = cp.Variable(self.state_dim * (pred_horizon + 1))
        self.__input_series = cp.Variable(self.input_dim * pred_horizon)

        self.__zero_terminal_set = zero_terminal_set
        self.__solver = solver

    @property
    def pred_horizon(self):
        return self.__pred_horizon

    @pred_horizon.setter
    def pred_horizon(self, value: int):
        if value <= 0:
            raise MPCException('The prediction horizon must be a positive integer!')

        self.__pred_horizon = value

    @property
    def state_prediction_series(self):
        return self.__state_series.value.reshape(self.__pred_horizon + 1, self.state_dim).T

    @property
    def input_prediction_series(self):
        return self.__input_series.value.reshape(self.__pred_horizon, self.input_dim).T

    @property
    def input_ini(self):
        return self.__input_series[0:self.input_dim]

    @property
    def state_ini(self):
        return self.__state_series[0:self.state_dim]

    @property
    def real_time_state(self):
        return self.__real_time_state

    @real_time_state.setter
    def real_time_state(self, value: np.ndarray):
        if value.ndim != 1:
            raise MPCException('The input state must be a 1D array!')
        if value.size != self.state_dim:
            raise MPCException('The dimension of the input state is wrong!')

        self.__real_time_state.value = value

    @property
    def zero_terminal_set(self):
        return self.__zero_terminal_set

    @zero_terminal_set.setter
    def zero_terminal_set(self, value: bool):
        self.__zero_terminal_set = value

    @property
    def problem(self):
        raise MPCException('Attribute \'problem\' can be only realized in subclass')

    @property
    def solver(self):
        return self.__solver

    @solver.setter
    def solver(self, value):
        self.__solver = value

    def cal_terminal_set(self, state_set: poly.Polyhedron, input_set: poly.Polyhedron) -> poly.Polyhedron:
        # 在终端约束 Xf 内的一点 x 满足：
        # 1. 当采用控制律 u = Kx 时，状态约束和输入约束均满足 -- 这一条件描述的集合为 X 与 U @ K 的交集，集合与矩阵的乘法解释请参考文件poly
        # 2. 下一时刻的状态 x+ = A_k @ x 仍属于 Xf -- 这一条件描述的集合 set 被包含于 set @ A_k
        # 若设置终端约束集合为原点，则生成一个边长为0的单位立方体，否则计算最大的满足上述条件的集合
        if self.__zero_terminal_set:
            terminal_set = poly.UnitCube(self.state_dim, 0)
        else:
            set_k = state_set & (input_set @ self.k)
            terminal_set = copy.deepcopy(set_k)
            a_k = self.a - self.b @ self.k

            while True:
                set_k = set_k @ a_k
                terminal_set_next = terminal_set & set_k

                if terminal_set.belongs_to(terminal_set_next):
                    break

                terminal_set = terminal_set_next

        return terminal_set

    def construct_problem(self, initial_constraints: cp.constraints.Constraint, state_set: poly.Polyhedron,
                          input_set: poly.Polyhedron, terminal_set: poly.Polyhedron):
        cost = 0
        state_k = self.__state_series[0:self.state_dim]

        # 对于初始状态的约束
        constraints = [initial_constraints]

        for k in range(self.__pred_horizon):
            input_k = self.__input_series[k * self.input_dim:(k + 1) * self.input_dim]

            # l(x, u) = x.T @ Q @ x + u.T @ R @ u
            cost = cost + (state_k @ self.q @ state_k + input_k @ self.r @ input_k) / 2

            # x^+ = A @ x + B @ u
            state_k_next = self.__state_series[(k + 1) * self.state_dim:(k + 2) * self.state_dim]
            constraints.append(cp.Zero(state_k_next - self.a @ state_k - self.b @ input_k))

            # x in X, u in U
            constraints.append(cp.NonPos(state_set(state_k)))
            constraints.append(cp.NonPos(input_set(input_k)))

            state_k = state_k_next

        # 另一种构建优化问题的思路，只把输入序列当作决策变量 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        #
        # constraints = []
        # state_k = self.__real_time_state
        #
        # for k in range(self.__pred_horizon):
        #     input_k = self.__input_series[k * self.__input_dim:(k + 1) * self.__input_dim]
        #
        #     cost = cost + (state_k @ self.q @ state_k + input_k @ self.r @ input_k)
        #     constraints.append(cp.NonPos(state_set(state_k)))
        #     constraints.append(cp.NonPos(input_set(input_k)))
        #
        #     state_k = self.a @ state_k + self.b @ input_k
        #
        # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = =

        cost = cost + (state_k @ self.p @ state_k) / 2
        if self.__zero_terminal_set:
            constraints.append(cp.Zero(state_k))
        else:
            constraints.append(cp.NonPos(terminal_set(state_k)))

        problem = cp.Problem(cp.Minimize(cost), constraints)

        return problem

    def cal_feasible_set(self, state_set: poly.Polyhedron, input_set: poly.Polyhedron,
                         terminal_set: poly.Polyhedron) -> poly.Polyhedron:
        # 这里先求出了M，C矩阵，于是知道了Xk = M*x + C*Uk，之后将约束条件转化为G*Xk <= h，再包含A_Uk*Uk <= b_Uk
        # 于是有
        # [G*M G*C ][x ]      [h   ]
        # [        ][  ]  <=  [    ]
        # [ 0  A_Uk][Uk]      [b_Uk]
        # a_bar, b_bar分别代表上面两个矩阵

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
        g = spl.block_diag(*[state_set.l_mat for _ in range(self.__pred_horizon)], terminal_set.l_mat)
        h = np.hstack((np.tile(state_set.r_vec, self.pred_horizon), terminal_set.r_vec))

        # 生成 A_Uk 和 b_Uk
        a_uk = spl.block_diag(*[input_set.l_mat for _ in range(self.__pred_horizon)])
        b_uk = np.tile(input_set.r_vec, self.pred_horizon)

        # 生成对应的零矩阵部分
        zero = np.zeros((a_uk.shape[0], self.state_dim))

        l_mat = np.block([[g @ m, g @ c], [zero, a_uk]])
        r_vec = np.hstack((h, b_uk))

        feasible_set = poly.Polyhedron(l_mat, r_vec)

        # 傅里叶-莫茨金消元法，将控制输入变量U消去
        feasible_set.fourier_motzkin_elimination(self.input_dim * self.__pred_horizon)

        return feasible_set


class MPC(MPCBase):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 state_set: poly.Polyhedron, input_set: poly.Polyhedron, zero_terminal_set=False, solver=cp.OSQP):
        super().__init__(a, b, q, r, pred_horizon, zero_terminal_set, solver)

        if not (self.state_dim == state_set.n_dim):
            raise MPCException('The dimensions of the state parameters do not match!')
        if not (self.input_dim == input_set.n_dim):
            raise MPCException('The dimensions of the input parameters do not match!')

        self.__state_set = state_set
        self.__input_set = input_set

        self.__terminal_set = self.cal_terminal_set(state_set, input_set)

        # 控制器内初始状态等于外部实际状态
        self.__initial_constraints = cp.Zero(self.state_ini - self.real_time_state)
        self.__problem = self.construct_problem(self.__initial_constraints,
                                                state_set,
                                                input_set,
                                                self.__terminal_set)

    @MPCBase.zero_terminal_set.setter
    def zero_terminal_set(self, value: bool):
        MPCBase.zero_terminal_set.fset(self, value)
        self.__problem = self.construct_problem(self.__initial_constraints,
                                                self.state_set,
                                                self.input_set,
                                                self.__terminal_set)

    @MPCBase.problem.getter
    def problem(self):
        return self.__problem

    @property
    def state_set(self):
        return self.__state_set

    @state_set.setter
    def state_set(self, value: poly.Polyhedron):
        if not self.state_dim == value.n_dim:
            raise MPCException('The dimension of the state set is Wrong!')
        self.__state_set = value

    @property
    def input_set(self):
        return self.__input_set

    @input_set.setter
    def input_set(self, value: poly.Polyhedron):
        if not self.input_dim == value.n_dim:
            raise MPCException('The dimension of the input set is Wrong!')
        self.__input_set = value

    @property
    def terminal_set(self):
        return self.__terminal_set

    @property
    def feasible_set(self):
        return self.cal_feasible_set(self.__state_set, self.__input_set, self.__terminal_set)

    def __call__(self, real_time_state: np.ndarray):
        if real_time_state.ndim != 1:
            raise MPCException('The input state must be a 1D array!')
        if real_time_state.size != self.state_dim:
            raise MPCException('The dimension of the input state is wrong!')

        self.real_time_state = real_time_state
        self.problem.solve(solver=self.solver)

        return self.input_ini.value
