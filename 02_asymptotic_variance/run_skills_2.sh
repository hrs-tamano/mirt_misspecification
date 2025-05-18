#!/bin/bash

if [ $#  -lt 3 ];
then
    echo "Usage: run_skills_2.sh <input_dir> <output_dir> <num_process>"
    echo ""
    echo "e.g., run_skills_2.sh data/skills_2 results/skills_2 60"
    exit
fi

DATA_DIR=$1
OUTPUT_DIR=$2
NUM_PROCESSES=$3

EXP_NUM_SAMPLES=10000
EXP_NUM_MODELS=30000
INF_NUM_SAMPLES=100000
INF_QUAD_PTS=30

# A setting just for  checking runnability
# EXP_NUM_SAMPLES=1000
# EXP_NUM_MODELS=5
# INF_NUM_SAMPLES=1000
# INF_QUAD_PTS=10

export OMP_NUM_THREADS=1
mkdir -p ${OUTPUT_DIR}

for i in `seq 20 49`
do
    echo "d_${i}_exp"
    python -u scripts/experiment_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} \
	   --num_samples ${EXP_NUM_SAMPLES} \
	   --num_models ${EXP_NUM_MODELS} \
	   --num_processes ${NUM_PROCESSES}
    
    echo "d_${i}_infer"
    python -u scripts/infer_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} \
	   --num_samples ${INF_NUM_SAMPLES} \
	   --num_quadpts ${INF_QUAD_PTS} \
	   --num_processes ${NUM_PROCESSES}
done


for i in `seq 20 49`
do
    echo "a_${i}_0_exp"
    python -u scripts/experiment_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} --skill_idx 0 --opt_slope \
	   --num_samples ${EXP_NUM_SAMPLES} \
	   --num_models ${EXP_NUM_MODELS} \
	   --num_processes ${NUM_PROCESSES}

    echo "a_${i}_0_infer"
    python -u scripts/infer_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} --skill_idx 0 --opt_slope \
	   --num_samples ${INF_NUM_SAMPLES} \
	   --num_quadpts ${INF_QUAD_PTS} \
	   --num_processes ${NUM_PROCESSES}
done


for i in `seq 20 49`
do
    echo "a_${i}_1_exp"
    python -u scripts/experiment_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} --skill_idx 1 --opt_slope \
	   --num_samples ${EXP_NUM_SAMPLES} \
	   --num_models ${EXP_NUM_MODELS} \
	   --num_processes ${NUM_PROCESSES}

    echo "a_${i}_1_infer"
    python -u scripts/infer_var.py \
	   ${DATA_DIR} ${OUTPUT_DIR} \
	   --item_idx ${i} --skill_idx 1 --opt_slope \
	   --num_samples ${INF_NUM_SAMPLES} \
	   --num_quadpts ${INF_QUAD_PTS} \
	   --num_processes ${NUM_PROCESSES}
done

python scripts/join_outputs.py ${OUTPUT_DIR} ${OUTPUT_DIR}
