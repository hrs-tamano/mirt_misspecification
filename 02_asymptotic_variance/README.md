# Experiment of Asymptotic Variance

## Prerequisites

- Python 3.9.13
- R 4.2.3
  - mirt 1.38.1
  - stats4 4.2.3
  - lattice 0.20.45

## How to Run

1. **Set Up Environment**

    ```bash
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -U pip
    $ pip install -r requirements.txt
    ```

2. **Generate Data**

    This command generates data in the `./data` directory.

    ```bash
    $ sh prepare_data.sh
    ```

3. **Run Script**

    Execute this script for the experiment involving two skills.

    ```bash
    $ sh run_skills_2.sh data/skills_2 results/skills_2 <num_process>
    ```

    This script runs in parallel with `<num_process>` processes. We used a machine with 64 cores, and it took several days to complete.

    Result files (`difficulty.csv` and `slope.csv`) will be generated in the `results/skills_2` directory.

    The command for three skills is as follows:

    ```bash
    $ sh run_skills_3.sh data/skills_3 results/skills_3 <num_process>
    ```

4. **Visualize Results**

    Run `Figures_tables.ipynb` in Jupyter Notebook.

    Note: Because random seeds are not fixed in this experiments, the results may vary slightly.
    However, the overall conclusions remain unchanged.