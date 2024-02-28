from .mpc import np, cp, poly, MPCException, MPCBase
# from .mpc import npl


class TubeBasedMPC(MPCBase):
    def __init__(self, a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, pred_horizon: int,
                 state_set: poly.Polyhedron, input_set: poly.Polyhedron, noise_set: poly.Polyhedron,
                 zero_terminal_set=False, solver=cp.PIQP):
        super().__init__(a, b, q, r, pred_horizon, zero_terminal_set, solver)

        if not (self.state_dim == noise_set.n_dim):
            raise MPCException('The dimension of the noise set do not match the state dimension!')

        self.__noise_set = noise_set
        self.__disturbance_invariant_set = self.cal_disturbance_invariant_set()

        self.__tightened_state_set = state_set - self.__disturbance_invariant_set
        self.__tightened_input_set = input_set - self.k @ self.__disturbance_invariant_set

        self.__terminal_set = self.cal_terminal_set(self.__tightened_state_set, self.__tightened_input_set)

        # 控制器内初始状态在以外部实际状态为中心的不变集里
        self.__initial_constraints = cp.NonPos(self.__disturbance_invariant_set(self.real_time_state - self.state_ini))
        self.__problem = self.construct_problem(self.__initial_constraints,
                                                self.__tightened_state_set,
                                                self.__tightened_input_set,
                                                self.__terminal_set)

    def __call__(self, real_time_state: np.ndarray) -> np.ndarray:
        self.real_time_state = real_time_state
        self.problem.solve(solver=self.solver)

        return self.input_ini.value - self.k @ (real_time_state - self.state_ini.value)

    @MPCBase.zero_terminal_set.setter
    def zero_terminal_set(self, value: bool) -> None:
        if self.zero_terminal_set != value:
            MPCBase.zero_terminal_set.fset(self, value)
            self.__terminal_set = self.cal_terminal_set(self.__tightened_state_set, self.__tightened_input_set)
            self.__problem = self.construct_problem(self.__initial_constraints,
                                                    self.__tightened_state_set,
                                                    self.__tightened_input_set,
                                                    self.__terminal_set)

    @property
    def noise_set(self) -> poly.Polyhedron:
        return self.__noise_set

    @noise_set.setter
    def noise_set(self, value: poly.Polyhedron) -> None:
        if not (self.state_dim == value.n_dim):
            raise MPCException('The dimension of the noise set do not match the state dimension!')

        self.__noise_set = value

    @property
    def disturbance_invariant_set(self) -> poly.Polyhedron:
        return self.__disturbance_invariant_set

    @property
    def tightened_state_set(self) -> poly.Polyhedron:
        return self.__tightened_state_set

    @property
    def tightened_input_set(self) -> poly.Polyhedron:
        return self.__tightened_input_set

    @property
    def terminal_set(self) -> poly.Polyhedron:
        return self.__terminal_set

    @property
    def feasible_set(self) -> poly.Polyhedron:
        return self.cal_feasible_set(self.__tightened_state_set, self.__tightened_input_set, self.__terminal_set)

    @MPCBase.problem.getter
    def problem(self) -> cp.Problem:
        return self.__problem

    def cal_disturbance_invariant_set(self, alpha=0.2, epsilon=0.001) -> poly.Polyhedron:
        alp = alpha

        # 由于多次给集合左乘 A_k，且 A_k 可逆，可以提前求好 A_k 的逆并在下面的 计算 1、计算 2 中右乘 A_k 的逆，这里为了方便理解，没有这么做
        # a_k_inv = npl.inv(self.a - self.b @ self.k)
        a_k = self.a - self.b @ self.k

        a_k_s = np.eye(self.state_dim)
        a_k_s_noise_set = self.__noise_set
        sum_a_k_s_noise_set = self.__noise_set
        alpha_noise_set = alp * self.__noise_set

        while True:
            if a_k_s_noise_set.belongs_to(alpha_noise_set):
                break

            a_k_s = a_k @ a_k_s

            # 计算 1 - - - - - - - - - - - - - - - - - - - #
            # a_k_s_noise_set = a_k_s_noise_set @ a_k_inv
            a_k_s_noise_set = a_k @ a_k_s_noise_set
            # - - - - - - - - - - - - - - - - - - - - - - #

            sum_a_k_s_noise_set = sum_a_k_s_noise_set + a_k_s_noise_set

        alpha_array = np.zeros(self.__noise_set.n_edges)
        for i in range(self.__noise_set.n_edges):
            alpha_array[i] = poly.support_fun(self.__noise_set.l_mat[i, :], a_k_s_noise_set) / a_k_s_noise_set.r_vec[i]

        alp = np.max(alpha_array)

        f_alpha_s_set = sum_a_k_s_noise_set / (1 - alp)

        a_k_n = a_k_s
        a_k_n_noise_set = a_k_s_noise_set
        sum_a_k_n_noise_set = sum_a_k_s_noise_set
        # 这里用一个单位球的内接超正方体代替单位球
        unit_cube = poly.UnitCube(self.state_dim, 2 * epsilon / np.sqrt(self.state_dim))

        while True:
            a_k_n = a_k @ a_k_n

            if a_k_n_noise_set.belongs_to(unit_cube):
                break

            # 计算 2 - - - - - - - - - - - - - - - - - - - #
            # a_k_n_noise_set = a_k_n_noise_set @ a_k_inv
            a_k_n_noise_set = a_k @ a_k_n_noise_set
            # - - - - - - - - - - - - - - - - - - - - - - #

            sum_a_k_n_noise_set = sum_a_k_n_noise_set + a_k_n_noise_set

        disturbance_invariant_set = a_k_n @ f_alpha_s_set + sum_a_k_n_noise_set

        return disturbance_invariant_set
