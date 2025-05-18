import os
import numpy as np
import pandas as pd
import argparse
import functools
import time
from multiprocessing import Pool
import mirt
from data import gen_ncm_data


def train_cm_par(
    num_samples: int,
    num_models: int,
    num_processes: int,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    option_str: str
):    
    with Pool(processes=num_processes) as p:
        func = functools.partial(
            train_cm,
            option_str=option_str,
            num_samples=num_samples,
            A=A,
            B=B,
            Q=Q
        )
        results = p.map(func, range(num_models))
        
    return results


def train_cm(idx: int, option_str: str, num_samples: int, A, B, Q):
    Y, Z = gen_ncm_data(num_samples, A, B, Q)
    Y_df = pd.DataFrame(Y)
    est_A, est_b, est_Z = mirt.fit_cm(Y_df, option_str, Q)
    return est_A, est_b, est_Z


def main(args):
    stime = time.time()
    data_dir = args.data_dir
    output_dir = args.output_dir

    A_true = np.load(os.path.join(data_dir, "A.npy"))
    B_true = np.load(os.path.join(data_dir, "B.npy"))
    Q = np.load(os.path.join(data_dir, "Q.npy"))

    est_A0 = np.load(os.path.join(data_dir, "est_A0.npy"))
    est_b0 = np.load(os.path.join(data_dir, "est_b0.npy"))

    item_idx = args.item_idx
    skill_idx = args.skill_idx

    q_mat_option = mirt.make_q_mat_option_str(Q)
    if args.opt_slope:
        opt_option = mirt.make_one_slope_param_opt_option_str(item_idx, skill_idx, est_A0, est_b0)
    else:
        opt_option = mirt.make_one_difficulty_param_opt_option_str(item_idx, est_A0, est_b0)

    num_samples = args.num_samples
    num_models = args.num_models
    num_processes = args.num_processes
    opt_slope = args.opt_slope

    results = train_cm_par(
        num_samples,
        num_models,
        num_processes,
        A_true,
        B_true,
        Q,
        q_mat_option + "\n" + opt_option
    )

    est_A_results = [res[0] for res in results]
    est_b_results = [res[1] for res in results]
    est_Z_results = [res[2] for res in results]

    est_As = np.array(est_A_results)
    est_bs = np.array(est_b_results)
    est_Zs = np.array(est_Z_results)

    if args.opt_slope:
        rel_est_As = (est_As - est_A0) * np.sqrt(num_samples)
        var_a = np.var(rel_est_As[:, item_idx, skill_idx])
        with open(os.path.join(output_dir, f"output_exp_a_{item_idx}_{skill_idx}.txt"), "w") as f:
            f.write(f"a,{item_idx},{skill_idx},{var_a}\n")
        print(f"var_a, {item_idx}, {skill_idx}, {var_a}")
    else:
        rel_est_bs = (est_bs - est_b0) * np.sqrt(num_samples)
        var_b = np.var(rel_est_bs[:, item_idx])
        with open(os.path.join(output_dir, f"output_exp_b_{item_idx}.txt"), "w") as f:
            f.write(f"b,{item_idx},{var_b}\n")
        print(f"var_b, {item_idx}, {var_b}")

    etime = time.time()
    print(f"done: {etime - stime:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--item_idx', type=int, default=0)
    parser.add_argument('--skill_idx', type=int, default=0)
    parser.add_argument("--opt_slope", action="store_true")
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--num_models", type=int, default=10_000)
    parser.add_argument("--num_processes", type=int, default=1)
    main(parser.parse_args())


