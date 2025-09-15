#!/bin/bash

python eval.py \
    --config_path /hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/configs/dynaclr/dynaclr_20250901_all.yml \
    --num_samples 20000 \
    --run_predict False \
    --run_ann_data True
