import os
import argparse
import numpy as np
from scipy.special import logit, expit
from typing import Tuple, Optional


def to_half_point(a: np.ndarray, b: np.ndarray):
    return np.array([
        b[0] + logit(np.sqrt(0.5)) / a[0],
        b[1] + logit(np.sqrt(0.5)) / a[1],
    ])


def get_b_from_half_point(a: np.ndarray, b: np.ndarray):
    return np.array([
        b[0] - logit(np.sqrt(0.5)) / a[0],
        b[1] - logit(np.sqrt(0.5)) / a[1],
    ])


def gen_ncm_data(
    num_users: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    num_problems, num_skills = Q.shape
    Z = np.random.normal(0.0, 1.0, size=(num_users, num_skills))

    Z_3d = np.tile(Z, reps=(num_problems, 1, 1)).transpose(1, 0, 2)
    B_3d = np.tile(B, reps=(num_users, 1, 1))
    A_3d = np.tile(A, reps=(num_users, 1, 1))
    Q_3d = np.tile(Q, reps=(num_users, 1, 1))

    U_3d = (Z_3d - B_3d) * A_3d
    P_3d = expit(U_3d)
    P_3d[Q_3d == 0.0] = 1.0
    P = np.prod(P_3d, axis=2)

    Y_1d = np.random.binomial(n=1, p=P.reshape(-1))
    Y = Y_1d.reshape(num_users, num_problems)

    return Y, Z


def gen_item_params(num_problems: int):

    Q = np.ones((num_problems, 2))
    Q[:10, 1] = 0
    Q[10:20, 0] = 0

    A = np.random.lognormal(0.2, 0.2, size=(num_problems, 2))
    
    B = np.zeros((num_problems, 2))
    B[:10, 0] = np.linspace(-2.5, 2.5, 10)
    B[10:20, 1] = np.linspace(-2.5, 2.5, 10)

    for i, b1 in enumerate(np.linspace(-2.5, 2.5, 10)):
        for j, b2 in enumerate(np.linspace(-2.5, 2.5, 10)):
            B[20 + i + j*10, :] = get_b_from_half_point(A[20 + i + j*10, :], [b1, b2])

    return A, B, Q
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    np.random.seed(100)
    
    num_problems = 120
    num_users = 1_000_000
    
    A, B, Q = gen_item_params(num_problems)
    np.save(os.path.join(args.output_dir, "A.npy"), A)
    np.save(os.path.join(args.output_dir, "B.npy"), B)
    np.save(os.path.join(args.output_dir, "Q.npy"), Q)

    Y0, Z0 = gen_ncm_data(num_users, A, B, Q)
    np.save(os.path.join(args.output_dir, "Y0.npy"), Y0)
    np.save(os.path.join(args.output_dir, "Z0.npy"), Z0)


