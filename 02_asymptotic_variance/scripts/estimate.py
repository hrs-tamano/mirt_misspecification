import numpy as np
import functools
from scipy.special import expit
from scipy.stats import multivariate_normal
from multiprocessing import Pool
from typing import Tuple

from gauss_hermite import integrate_with_gauss


def calc_q(y: float, u: float):
    return expit(u) if y == 1.0 else 1.0 - expit(u)


def calc_ln_q_ys(ys: np.ndarray, gamma: np.ndarray, A: np.ndarray, b: np.ndarray):
    ps = expit(A @ gamma + b)
    cond = (ys == 0)
    ps[cond] = (1.0 - ps[cond])
    cond_small = (ps < 1e-16)
    ps[cond_small] = 1e-16
    ln_q_ys = np.sum(np.log(ps))
    return ln_q_ys


def calc_rel_q_y(
        gamma_l: np.ndarray,
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        ln_q_base: float
):
    ln_q_y_l = calc_ln_q_ys(ys, gamma_l, A, b)
    return np.exp(ln_q_y_l - ln_q_base)


def calc_rel_der_q_y_der_b(
        gamma_l: np.ndarray,
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        ln_q_base: float
):
    u = A[h, :] @ gamma_l + b[h]
    ln_q_y_l = calc_ln_q_ys(ys, gamma_l, A, b)
    rel_q_y_l = np.exp(ln_q_y_l - ln_q_base)
    return (ys[h] - expit(u)) * rel_q_y_l


def calc_rel_der_q_y_der_a(
        gamma_l: np.ndarray,
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        s: int,
        ln_q_base: float
):
    u = A[h, :] @ gamma_l + b[h]
    ln_q_y_l = calc_ln_q_ys(ys, gamma_l, A, b)
    rel_q_y_l = np.exp(ln_q_y_l - ln_q_base)
    return (ys[h] - expit(u)) * rel_q_y_l * gamma_l[s]


def calc_der_ln_q_der_b(
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        num_quadpts: int = 10
):
    num_items, num_dim = A.shape

    zero = np.zeros(num_dim)
    I = np.eye(num_dim)

    ln_q_base = calc_ln_q_ys(ys, zero, A, b)
    func1 = functools.partial(calc_rel_q_y, ys=ys, A=A, b=b, ln_q_base=ln_q_base)
    func2 = functools.partial(calc_rel_der_q_y_der_b, ys=ys, A=A, b=b, h=h, ln_q_base=ln_q_base)

    rel_q_y = integrate_with_gauss(zero, I, func1, num_quadpts)
    rel_der_q_y = integrate_with_gauss(zero, I, func2, num_quadpts)

    return rel_der_q_y / rel_q_y


def calc_der_ln_q_der_a(
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        s: int,
        num_quadpts: int = 10
):
    num_items, num_dim = A.shape

    zero = np.zeros(num_dim)
    I = np.eye(num_dim)

    ln_q_base = calc_ln_q_ys(ys, zero, A, b)
    func1 = functools.partial(calc_rel_q_y, ys=ys, A=A, b=b, ln_q_base=ln_q_base)
    func2 = functools.partial(calc_rel_der_q_y_der_a, ys=ys, A=A, b=b, h=h, s=s, ln_q_base=ln_q_base)

    rel_q_y = integrate_with_gauss(zero, I, func1, num_quadpts)
    rel_der_q_y = integrate_with_gauss(zero, I, func2, num_quadpts)

    return rel_der_q_y / rel_q_y


def calc_rel_der2_q_y_der2_b(
        gamma_l: np.ndarray,
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        ln_q_base: float
):
    u = A[h, :] @ gamma_l + b[h]
    ln_q_y_l = calc_ln_q_ys(ys, gamma_l, A, b)
    rel_q_y_l = np.exp(ln_q_y_l - ln_q_base)
    v = (ys[h] ** 2 - 2.0 * ys[h] * expit(u) - expit(u) + 2.0 * expit(u) ** 2)
    return v * rel_q_y_l


def calc_rel_der2_q_y_der2_a(
        gamma_l: np.ndarray,
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        s: int,
        ln_q_base: float
):
    u = A[h, :] @ gamma_l + b[h]
    ln_q_y_l = calc_ln_q_ys(ys, gamma_l, A, b)
    rel_q_y_l = np.exp(ln_q_y_l - ln_q_base)
    v = (ys[h] ** 2 - 2.0 * ys[h] * expit(u) - expit(u) + 2.0 * expit(u) ** 2)
    return v * rel_q_y_l * (gamma_l[s] ** 2)


def calc_der2_q_der2_b_div_q(
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        num_quadpts: int = 10
):
    num_items, num_dim = A.shape

    zero = np.zeros(num_dim)
    I = np.eye(num_dim)

    ln_q_base = calc_ln_q_ys(ys, zero, A, b)

    func1 = functools.partial(calc_rel_q_y, ys=ys, A=A, b=b, ln_q_base=ln_q_base)
    func2 = functools.partial(calc_rel_der2_q_y_der2_b, ys=ys, A=A, b=b, h=h, ln_q_base=ln_q_base)

    rel_q_y = integrate_with_gauss(zero, I, func1, num_quadpts)
    rel_der2_q_y = integrate_with_gauss(zero, I, func2, num_quadpts)

    return rel_der2_q_y / rel_q_y


def calc_der2_q_der2_a_div_q(
        ys: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        s: int,
        num_quadpts: int = 10
):
    num_items, num_dim = A.shape

    zero = np.zeros(num_dim)
    I = np.eye(num_dim)

    ln_q_base = calc_ln_q_ys(ys, zero, A, b)

    func1 = functools.partial(calc_rel_q_y, ys=ys, A=A, b=b, ln_q_base=ln_q_base)
    func2 = functools.partial(calc_rel_der2_q_y_der2_a, ys=ys, A=A, b=b, h=h, s=s, ln_q_base=ln_q_base)

    rel_q_y = integrate_with_gauss(zero, I, func1, num_quadpts)
    rel_der2_q_y = integrate_with_gauss(zero, I, func2, num_quadpts)

    return rel_der2_q_y / rel_q_y


def estimate_b_IJ(
        Y: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        num_processes: int,
        num_quadpts: int = 10
):
    with Pool(processes=num_processes) as p:
        func = functools.partial(calc_der_ln_q_der_b, A=A, b=b, h=h, num_quadpts=num_quadpts)
        results = p.map(func, Y)

    J = np.mean([v ** 2 for v in results])

    with Pool(processes=num_processes) as p:
        func = functools.partial(calc_der2_q_der2_b_div_q, A=A, b=b, h=h, num_quadpts=num_quadpts)
        results = p.map(func, Y)

    I = J + np.mean([-v for v in results])
    return I, J


def estimate_a_IJ(
        Y: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        s: int,
        num_processes: int,
        num_quadpts: int = 10
):
    with Pool(processes=num_processes) as p:
        func = functools.partial(calc_der_ln_q_der_a, A=A, b=b, h=h, s=s, num_quadpts=num_quadpts)
        results = p.map(func, Y)

    J = np.mean([v ** 2 for v in results])

    with Pool(processes=num_processes) as p:
        func = functools.partial(calc_der2_q_der2_a_div_q, A=A, b=b, h=h, s=s, num_quadpts=num_quadpts)
        results = p.map(func, Y)

    I = J + np.mean([-v for v in results])
    return I, J


def estimate_der_ln_q(
        Y: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        h: int,
        num_processes: int,
        num_quadpts: int = 10
):
    with Pool(processes=num_processes) as p:
        func = functools.partial(calc_der_ln_q_der_b, A=A, b=b, h=h, num_quadpts=num_quadpts)
        results = p.map(func, Y)

    return np.sum(results)
