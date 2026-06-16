#!/bin/bash --login
# Direct high-resolution graph-generation launcher for Lyris.
# Submit from this directory:
#   cd ./aifs/launchers/SLURM/direct
#   sbatch high-res.generate-graph.slurm

#SBATCH --job-name=high-res.generate-graph
#SBATCH --nodes=1
#SBATCH --partition=gb300
#SBATCH --account=coreai_devtech_all
#SBATCH --segment=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH -o ./logs/%x-%j.slurm.out

set -euo pipefail

script_dir=$(pwd -P)
cd "$script_dir"

env \
  benchmark_command=highres-generate-graph \
  batch_limit=1 \
  max_epochs=1 \
  run_prefix=generate-graph \
  bash high-res.slurm
