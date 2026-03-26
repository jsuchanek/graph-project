#!/bin/bash
#SBATCH --job-name=precompute_edges
#SBATCH --output=logs/precompute_%j.out
#SBATCH --error=logs/precompute_%j.err
#SBATCH --time=64:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G


module purge
module load anaconda3
cd $SLURM_SUBMIT_DIR

echo "Running on node: $(hostname)"
echo "Python: $(which python)"
conda run -n graphs-env python -u precompute/precompute_edge_features.py

