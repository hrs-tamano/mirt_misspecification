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
        "optimizer": "'BFGS'",
    }

    fit_options_str = ", ".join([f"{k}={v}" for k, v in fit_options.items()])

    ro.r.assign("df", rdf)
    ro.r.assign("option_str", option_str)
    ro.r("cm_model <- mirt.model(option_str)")
    ro.r(f"cm_res <- mirt(df, cm_model, {fit_options_str})")
    ro.r("cm_ab <- coef(cm_res)")
    ro.r(f"cm_theta <- fscores(cm_res, method='MAP')")

    est_A = np.zeros((num_problems, num_skills))
    est_b = np.zeros(num_problems)
    for i in range(num_problems):
        for k in range(num_skills):
            est_A[i, k] = ro.r(f"cm_ab[[{i + 1}]][{k + 1}]")[0]
        est_b[i] = ro.r(f"cm_ab[[{i + 1}]][{num_skills + 1}]")[0]

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
