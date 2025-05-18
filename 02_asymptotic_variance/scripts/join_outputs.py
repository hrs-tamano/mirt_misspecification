import os
import argparse
import pandas as pd


def main(input_dir: str, output_dir: str):

    inf_a_files = [f for f in os.listdir(input_dir) if f.startswith("output_a_")]
    inf_b_files = [f for f in os.listdir(input_dir) if f.startswith("output_b_")]
    exp_a_files = [f for f in os.listdir(input_dir) if f.startswith("output_exp_a_")]
    exp_b_files = [f for f in os.listdir(input_dir) if f.startswith("output_exp_b_")]

    inf_a_cols = ["a_or_b", "Problem", "Dim", "J/I/I", "1/I"]
    inf_b_cols = ["a_or_b", "Problem", "J/I/I", "1/I"]
    exp_a_cols = ["a_or_b", "Problem", "Dim", "ExpVar"]
    exp_b_cols = ["a_or_b", "Problem", "ExpVar"]

    ## create df for infer and slope
    inf_a_dfs = []
    for f in inf_a_files:
        df = pd.read_csv(os.path.join(input_dir, f), names=inf_a_cols)
        inf_a_dfs.append(df)

    inf_a_df = pd.concat(inf_a_dfs, axis=0)
    inf_a_df.drop(columns=["a_or_b"], inplace=True)
    inf_a_df = inf_a_df.sort_values(["Problem", "Dim"], ignore_index=True)
    inf_a_df.head()

    ## create df for infer and difficulty
    inf_b_dfs = []
    for f in inf_b_files:
        df = pd.read_csv(os.path.join(input_dir, f), names=inf_b_cols)
        inf_b_dfs.append(df)

    inf_b_df = pd.concat(inf_b_dfs, axis=0)
    inf_b_df.drop(columns=["a_or_b"], inplace=True)
    inf_b_df = inf_b_df.sort_values(["Problem"], ignore_index=True)
    inf_b_df.head()

    ## create df for experiment and slope
    exp_a_dfs = []
    for f in exp_a_files:
        df = pd.read_csv(os.path.join(input_dir, f), names=exp_a_cols)
        exp_a_dfs.append(df)

    exp_a_df = pd.concat(exp_a_dfs, axis=0)
    exp_a_df.drop(columns=["a_or_b"], inplace=True)
    exp_a_df = exp_a_df.sort_values(["Problem", "Dim"], ignore_index=True)
    exp_a_df.head()

    ## create df for experiment and difficulty
    exp_b_dfs = []
    for f in exp_b_files:
        df = pd.read_csv(os.path.join(input_dir, f), names=exp_b_cols)
        exp_b_dfs.append(df)

    exp_b_df = pd.concat(exp_b_dfs, axis=0)
    exp_b_df.drop(columns=["a_or_b"], inplace=True)
    exp_b_df = exp_b_df.sort_values(["Problem"], ignore_index=True)
    exp_b_df.head()

    # join
    join_a_df = inf_a_df.merge(exp_a_df, on=["Problem", "Dim"])
    join_b_df = inf_b_df.merge(exp_b_df, on=["Problem"])
    
    join_a_df.to_csv(
        os.path.join(output_dir, f"slope.csv"),
        index=False
    )
    join_b_df.to_csv(
        os.path.join(output_dir, f"difficulty.csv"),
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    main(
        args.input_dir,
        args.output_dir,
    )




