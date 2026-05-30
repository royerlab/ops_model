#!/bin/bash

#SBATCH --job-name=dynaclr_ops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=14G
#SBATCH --constraint="h100|h200"
#SBATCH --time=0-22:00:00
#SBATCH --output=./slurm_logs/dynaclr_ops_%j.log

# debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

module load anaconda/latest
conda activate ops-model

python_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/train.py"
config_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/bag_of_channels/fit_gene_n_marker.yml"

cat $config_file
echo "--------------------------------"
cat $python_file
echo "--------------------------------"

srun python $python_file --config_path $config_file