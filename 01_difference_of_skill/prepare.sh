#!/bin/sh
DATA_DIR=data

mkdir -p $DATA_DIR

echo "Generating data"
python scripts/gen_data.py $DATA_DIR

echo "Fitting model"
python scripts/fit_cm.py $DATA_DIR