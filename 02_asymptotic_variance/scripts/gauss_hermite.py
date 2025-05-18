import numpy as np
import itertools


def integrate_with_gauss(mu: np.ndarray, Sigma: np.ndarray, func, n: int = 10):

    x, w = np.polynomial.hermite.hermgauss(n)
    N = len(mu)
    const = np.pi**(-0.5*N)
    xn = np.array(list(itertools.product(*(x,)*N)))
    wn = np.prod(np.array(list(itertools.product(*(w,)*N))), 1)
    yn = 2.0**0.5*np.dot(np.linalg.cholesky(Sigma), xn.T).T + mu[None, :]

    return np.sum(wn * const * np.array(list(map(func, yn))))