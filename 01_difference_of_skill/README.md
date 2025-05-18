# Experiment of Differences in Estimated Skills

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

2. **Prepare Data and Model Parameters**

    This command generates data and model parameters in the `./data` directory.
    (It took about one hour in our environment.)

    ```bash
    $ sh prepare.sh
    ```

3. **Visualize Figures**

    Launch Jupyter and run the following notebooks: `Fig_6.ipynb`, `Fig_7.ipynb`, and `Fig_8.ipynb`.
    
    Note: Due to differences in random seeds used in the experiments, the results may vary slightly.
    However, the overall conclusions remain unchanged.