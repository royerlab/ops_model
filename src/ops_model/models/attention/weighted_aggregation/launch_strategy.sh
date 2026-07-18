#!/bin/bash
# Parameterized driver launcher.
#
# Usage:
#   sbatch --job-name=<strategy>_driver launch_strategy.sh <strategy>
#
# Example:
#   sbatch --job-name=v5_gko_driver launch_strategy.sh v5_gko
#   sbatch --job-name=set_accuracy_driver launch_strategy.sh set_accuracy
#
# The --job-name override on sbatch lets one script serve every strategy
# — no more per-strategy shell scripts to maintain.
#
#SBATCH --output=/home/gav.sturm/linked_folders/mydata/ops_mono/slurm_logs/v5_attn_driver/%j.out
#SBATCH --error=/home/gav.sturm/linked_folders/mydata/ops_mono/slurm_logs/v5_attn_driver/%j.err
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: sbatch --job-name=<strategy>_driver $0 <strategy> [--signal-set <set>]" >&2
    exit 1
fi

STRATEGY="$1"
shift
SIGNAL_SET="${1:-phase_only}"

cd /hpc/mydata/gav.sturm/ops_mono
source .venv/bin/activate 2>/dev/null || true

exec python -u -m ops_model.models.attention.weighted_aggregation.run_v3_pipeline_on_v4_attn_weighted \
    --attn-strategy "$STRATEGY" \
    --signal-set "$SIGNAL_SET" \
    --slurm \
    --slurm-partition gpu
