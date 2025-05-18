import numpy as np
from typing import Tuple, Optional
import datetime
from scipy.special import expit


def gen_ncm_data(
    num_users: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if random_seed is None:
        np.random.seed(int(datetime.datetime.now().strftime('%f')))
    else:
        np.random.seed(random_seed)

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
