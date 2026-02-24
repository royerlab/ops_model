#!/bin/bash

#SBATCH --job-name=dynaclr_ops_predict
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=15G

#SBATCH --time=0-3:00:00
#SBATCH --output=./slurm_logs/dynaclr_ops_%j.log

# debugging flags (optional)
# https://lightning.ai/docs/pytorch/stable/clouds/cluster_advanced.html
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

module load anaconda/latest
conda activate ops-model

python_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/predict.py"
config_file="/home/eduardo.hirata/repos/ops_model/experiments/models/dynaclr/phase_only/predict/predict.yml"

cat $config_file
echo "--------------------------------"
cat $python_file
echo "--------------------------------"

srun python $python_file --config_path $config_file