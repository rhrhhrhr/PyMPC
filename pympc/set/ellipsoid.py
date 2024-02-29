from .poly import np, npl, cp


class EllipsoidException(Exception):
    pass


class Ellipsoid(object):
    def __init__(self, p: np.ndarray, alpha: int or float, center: np.ndarray = None):
        try:
            _ = npl.cholesky(p)
        except npl.LinAlgError:
            raise EllipsoidException('The matrix of ellipsoid is not positive definite!')

        self.__p = p
        self.__n_dim = p.shape[0]
        self.__alpha = alpha

        self.__center = np.zeros(self.__n_dim) if center is None else center

    def __call__(self, point: np.ndarray or cp.Expression) -> np.ndarray or cp.Expression:
        return (point - self.__center) @ self.__p @ (point - self.__center)
