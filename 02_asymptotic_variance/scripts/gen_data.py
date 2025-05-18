import os
import argparse
import numpy as np
from data import gen_ncm_data


def gen_2skills_data(output_dir: str, seed: int):
    np.random.seed(seed)
    num_users = 1_000_000
    num_problems = 50
    num_skills = 2

    Q = np.ones((num_problems, num_skills))
    Q[:10, 1] = 0
    Q[10:20, 0] = 0

    A = np.random.lognormal(0.2, 0.2, size=(num_problems, num_skills))

    B = np.zeros((num_problems, num_skills))
    B[:10, 0] = np.linspace(-3.0, 3.0, 10)
    B[10:20, 1] = np.linspace(-3.0, 3.0, 10)
    B[20:, :] = np.random.normal(-1.0, np.sqrt(1.5), size=(30, 2))

    np.save(os.path.join(output_dir, "A.npy"), A)
    np.save(os.path.join(output_dir, "B.npy"), B)
    np.save(os.path.join(output_dir, "Q.npy"), Q)

    Y0, Z0 = gen_ncm_data(num_users, A, B, Q, seed)
    np.save(os.path.join(output_dir, "Y0.npy"), Y0)
    np.save(os.path.join(output_dir, "Z0.npy"), Z0)

    Y1, Z1 = gen_ncm_data(num_users, A, B, Q, seed + 1000)
    np.save(os.path.join(output_dir, "Y1.npy"), Y1)
    np.save(os.path.join(output_dir, "Z1.npy"), Z1)


def gen_3skills_data(output_dir: str, seed: int):
    np.random.seed(seed)
    num_users = 1_000_000
    num_problems = 50
    num_skills = 3

    Q = np.zeros((num_problems, num_skills))
    # 1 skill
    Q[:5, 0] = 1
    Q[5:10, 1] = 1
    Q[10:15, 2] = 1
    # 2 skill
    Q[15:20, [0, 1]] = 1
    Q[20:25, [1, 2]] = 1
    Q[25:30, [0, 2]] = 1
    # 3 skill
    Q[30:, :] = 1

    A = np.random.lognormal(0.2, 0.2, size=(num_problems, num_skills))
    A *= Q

    B = np.zeros((num_problems, num_skills))
    B[:5, 0] = np.linspace(-2.0, 2.0, 5)
    B[5:10, 1] = np.linspace(-2.0, 2.0, 5)
    B[10:15, 2] = np.linspace(-2.0, 2.0, 5)
    B[15:30, :] = np.random.normal(-1.0, np.sqrt(1.5), size=(15, num_skills))
    B[30:50, :] = np.random.normal(-1.5, np.sqrt(1.5), size=(20, num_skills))
    B *= Q
    
    np.save(os.path.join(output_dir, "A.npy"), A)
    np.save(os.path.join(output_dir, "B.npy"), B)
    np.save(os.path.join(output_dir, "Q.npy"), Q)
    
    Y0, Z0 = gen_ncm_data(num_users, A, B, Q, seed)
    np.save(os.path.join(output_dir, "Y0.npy"), Y0)
    np.save(os.path.join(output_dir, "Z0.npy"), Z0)
    
    Y1, Z1 = gen_ncm_data(num_users, A, B, Q, seed + 1000)
    np.save(os.path.join(output_dir, "Y1.npy"), Y1)
    np.save(os.path.join(output_dir, "Z1.npy"), Z1)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir', type=str)
    parser.add_argument('num_skills', type=int)
    parser.add_argument('seed', type=int)    
    args = parser.parse_args()
    
    if args.num_skills == 2:
        gen_2skills_data(args.output_dir, args.seed)
    elif args.num_skills == 3:
        gen_3skills_data(args.output_dir, args.seed)
    else:
        raise ValueError(f"num_skills must be 2 or 3.")

