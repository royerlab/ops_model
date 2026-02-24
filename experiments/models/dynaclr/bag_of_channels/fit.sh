#!/bin/bash

#SBATCH --job-name=dynaclr_ops
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=12G
#SBATCH --constraint="a100_80|h100|h200"

#SBATCH --time=0-22:00:00
#SBATCH --output=./slurm_logs/dynaclr_ops_%j.log

# debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load anaconda/latest
conda activate ops-model

python_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/train.py"
config_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/bag_of_channels/train_bagofchannels.yml"

cat $config_file
echo "--------------------------------"
cat $python_file
echo "--------------------------------"

srun python $python_file --config_path $config_file