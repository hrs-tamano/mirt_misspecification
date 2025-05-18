import argparse
import os

import numpy as np
import pandas as pd
import mirt


def main(args):
    Y0 = np.load(os.path.join(args.data_dir, "Y0.npy"))
    Q = np.load(os.path.join(args.data_dir, "Q.npy"))
    option_str = mirt.make_q_mat_option_str(Q)

    pdf = pd.DataFrame(Y0)
    est_A, est_b, est_Z = mirt.fit_cm(pdf, option_str, Q)

    np.save(os.path.join(args.data_dir, "est_A0.npy"), est_A)
    np.save(os.path.join(args.data_dir, "est_b0.npy"), est_b)
    np.save(os.path.join(args.data_dir, "est_Z0.npy"), est_Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    main(parser.parse_args())
    print("done")