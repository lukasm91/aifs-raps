#!/bin/bash --login
# Thin high-resolution launcher for the prebuilt AIFS-RAPS container.
# Submit from this directory:
#   sbatch high-res.slurm
# Optional editable anemoi-core checkout:
#   sbatch --export=ALL,anemoi_core_host=/path/to/anemoi-core high-res.slurm
# Optional external dataset:
#   sbatch --export=ALL,data=/path/to/aifs-benchmark-4km-2years-synthetic.zarr high-res.slurm
# Optional benchmark overrides:
#   sbatch -N 16 --export=ALL,ens_size=4,batch_limit=90,max_epochs=2,run_prefix=test high-res.slurm

#SBATCH --job-name=high-res
#SBATCH --nodes=64
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
container=${container:-${AIFS_IMAGE}}
container_name=${container_name:-aifs-highres}
benchmark_command=${benchmark_command:-highres}
anemoi_core_host=${anemoi_core_host:-${ANEMOI_CORE_HOST:-}}
data=${data:-}
num_tasks=$((SLURM_NNODES * 4))
graph_dir_host="$script_dir/graphs"
mounts="$script_dir:/work,$graph_dir_host:/graphs"

if [[ -n "$data" ]]; then
  if [[ ! -d "$data" ]]; then
    echo "data does not exist or is not a directory: $data" >&2
    exit 1
  fi
  if [[ "$data" != *.zarr ]]; then
    echo "data must point to the actual high-res .zarr directory: $data" >&2
    exit 1
  fi
  data=$(cd "$data" && pwd -P)
  mounts="$mounts,$data:/data/high-res/aifs-benchmark-4km-2years-synthetic.zarr:ro"
fi

if [[ -n "$anemoi_core_host" ]]; then
  if [[ ! -d "$anemoi_core_host" ]]; then
    echo "anemoi_core_host does not exist or is not a directory: $anemoi_core_host" >&2
    exit 1
  fi
  anemoi_core_host=$(cd "$anemoi_core_host" && pwd -P)
  mounts="$mounts,$anemoi_core_host:/opt/anemoi-core"
fi

mkdir -p "$script_dir/logs" "$graph_dir_host"
export SLURM_OVERLAP=1

srun \
  -n "$num_tasks" \
  --mpi=pmix \
  --container-image="$container" \
  --container-mounts "$mounts" \
  --container-name "$container_name" \
  --container-workdir="/work" \
  aifs-benchmark "$benchmark_command"
