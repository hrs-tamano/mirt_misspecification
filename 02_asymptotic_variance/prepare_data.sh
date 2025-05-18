#!/bin/sh

DATA_DIR="data"
RAND_SEED=0

for num_skill in 2 3
do
    OUTPUT_DIR="${DATA_DIR}/skills_${num_skill}"
    mkdir -p $OUTPUT_DIR
    echo "generating ${OUTPUT_DIR}"
    python script/gen_data.py ${OUTPUT_DIR} ${num_skill} ${RAND_SEED}
    echo "fitting a compensatory model"
    python script/fit_cm.py ${OUTPUT_DIR}
done

