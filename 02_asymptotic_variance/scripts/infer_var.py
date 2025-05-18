import os
import numpy as np
import pandas as pd
import argparse
import time
from estimate import estimate_a_IJ, estimate_b_IJ


def main(args):
    stime = time.time()
    data_dir = args.data_dir
    output_dir = args.output_dir

    est_A0 = np.load(os.path.join(data_dir, "est_A0.npy"))
    est_b0 = np.load(os.path.join(data_dir, "est_b0.npy"))
    Q = np.load(os.path.join(data_dir, "Q.npy"))
    Y1 = np.load(os.path.join(data_dir, "Y1.npy"))

    item_idx = args.item_idx
    skill_idx = args.skill_idx
    num_samples = args.num_samples

    if args.opt_slope:
        I, J = estimate_a_IJ(
            Y1[:num_samples],
            est_A0,
            est_b0,
            item_idx,
            skill_idx,
            num_processes=args.num_processes,
            num_quadpts=args.num_quadpts
        )
        with open(os.path.join(output_dir, f"output_a_{item_idx}_{skill_idx}.txt"), "w") as f:
            f.write(f"a,{item_idx},{skill_idx},{J / I / I},{1 / I}\n")
        print(f"a, {item_idx}, {skill_idx}, {J / I / I}, {1 / I}")
    else:
        I, J = estimate_b_IJ(
            Y1[:num_samples],
            est_A0,
            est_b0,
            item_idx,
            num_processes=args.num_processes,
            num_quadpts=args.num_quadpts
        )
        with open(os.path.join(output_dir, f"output_b_{item_idx}.txt"), "w") as f:
            f.write(f"b,{item_idx},{J / I / I},{1 / I}\n")
        print(f"b, {item_idx}, {J / I / I}, {1 / I}")

    etime = time.time()
    print(f"done: {etime - stime:.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--item_idx', type=int, default=0)
    parser.add_argument('--skill_idx', type=int, default=0)
    parser.add_argument("--opt_slope", action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--num_quadpts", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=1_000_000)

    main(parser.parse_args())


