#!/bin/bash
#SBATCH --job-name=set_acc_ebi_attn_driver
#SBATCH --output=/home/gav.sturm/linked_folders/mydata/ops_mono/slurm_logs/set_accuracy_attn_driver/%j.out
#SBATCH --error=/home/gav.sturm/linked_folders/mydata/ops_mono/slurm_logs/set_accuracy_attn_driver/%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
cd /hpc/mydata/gav.sturm/ops_mono
source .venv/bin/activate 2>/dev/null || true

exec python -u -m ops_model.models.attention.weighted_aggregation.run_v3_pipeline_on_v4_attn_weighted \
    --attn-strategy set_accuracy_ebi \
    --signal-set phase_only \
    --slurm \
    --slurm-partition gpu
