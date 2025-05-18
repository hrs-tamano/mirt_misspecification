import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
importr("mirt")


def fit_cm(pdf: pd.DataFrame, option_str: str, Q: np.ndarray):
    num_problems, num_skills = Q.shape
    num_users = pdf.shape[0]

    with (ro.default_converter + pandas2ri.converter).context():
        rdf = ro.conversion.get_conversion().py2rpy(pdf)

    fit_options = {
        "method": "'EM'",
        "itemtype": "'2PL'",
        "optimizer": "'NR'",
        "TOL": "0.000001",
        "technical": "list(NCYCLES=10000)",
    }
    fit_options_str = ", ".join([f"{k}={v}" for k, v in fit_options.items()])

    ro.r.assign("df", rdf)
    ro.r.assign("option_str", option_str)
    ro.r("cm_model <- mirt.model(option_str)")
    ro.r(f"cm_res <- mirt(df, cm_model, {fit_options_str})")
    ro.r("cm_ab <- coef(cm_res)")
    ro.r("cm_theta <- fscores(cm_res)")

    est_A = np.zeros((num_problems, num_skills))
    est_b = np.zeros(num_problems)
    for i in range(num_problems):
        for k in range(num_skills):
            est_A[i, k] = ro.r(f"cm_ab[[{i + 1}]][{k + 1}]")[0]  # a1,a2,...,ak,d

        est_b[i] = ro.r(f"cm_ab[[{i + 1}]][{num_skills + 1}]")[0]  # d

    est_Z = np.zeros((num_users, num_skills))
    for i in range(num_users):
        for k in range(num_skills):
            est_Z[i, k] = ro.r(f"cm_theta[{i + 1},{k + 1}]")[0]

    return est_A, est_b, est_Z


def make_q_mat_option_str(Q: np.ndarray):
    num_skills = Q.shape[1]
    f_list = []
    for k in range(num_skills):
        fk = ",".join([f"{i + 1}" for i in np.where(Q[:, k] == 1)[0]])
        f_list.append(f"F{k + 1} = {fk}")

    return "\n".join(f_list)


def make_one_slope_param_opt_option_str(
    h: int,  # item idx
    s: int,  # skill idx
    A: np.ndarray,
    b: np.ndarray,
):
    num_problems = b.shape[0]
    num_skills = A.shape[1]

    start_vals_a = "\n".join(
        [
            f"START = ({i + 1}, a{k + 1}, {A[i, k]})"
            for k in range(num_skills)
            for i in range(num_problems)
            if (i != h or k != s)
        ]
    )
    start_vals_d = "\n".join(
        [f"START = ({i + 1}, d, {b[i]})" for i in range(num_problems)]
    )

    fixed_a = "\n".join(
        [
            f"FIXED = ({i + 1}, a{k + 1})"
            for k in range(num_skills)
            for i in range(num_problems)
            if (i != h or k != s)
        ]
    )
    fixed_d = "\n".join(
        [f"FIXED = ({i + 1}, d)" for i in range(num_problems)]
    )

    option_str = "\n".join([
        start_vals_a,
        start_vals_d,
        fixed_a,
        fixed_d
    ])

    return option_str


def make_one_difficulty_param_opt_option_str(
    h: int,  # item idx
    A: np.ndarray,
    b: np.ndarray,
):
    num_problems = b.shape[0]
    num_skills = A.shape[1]

    start_vals_a = []
    for k in range(num_skills):
        start_vals_ak = [
            f"START = ({i + 1}, a{k + 1}, {A[i, k]})" for i in range(num_problems)
        ]
        start_vals_a += start_vals_ak

    fixed_a = []
    for k in range(num_skills):
        fixed_a += [f"FIXED = (1-{num_problems}, a{k + 1})"]

    start_vals_d = [
        f"START = ({i + 1}, d, {b[i]})" for i in range(num_problems) if i != h
    ]

    fixed_d = [
        f"FIXED = ({i + 1}, d)" for i in range(num_problems) if i != h
    ]

    option_str = "\n".join(
        start_vals_a + start_vals_d + fixed_a + fixed_d
    )

    return option_str
