#!/bin/bash

#SBATCH --job-name=dynaclr_vesuvius
#SBATCH --output=/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr_vesuvius/slurm_logs/%j.out
#SBATCH --error=/hpc/projects/intracellular_dashboard/ops/models/logs/dynaclr_vesuvius/slurm_logs/%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

module load mamba

mamba activate /hpc/mydata/alexander.hillsley/ops_env

srun python /hpc/mydata/alexander.hillsley/ops/ops_analysis/ops_analysis/model/vesuvius/train/train_dynaclr.py