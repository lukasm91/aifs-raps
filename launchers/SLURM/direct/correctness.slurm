#!/bin/bash --login
# Thin correctness launcher for the prebuilt AIFS-RAPS container.
# Submit from this directory:
#   sbatch --export=ALL,data=/path/to/aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr correctness.slurm
# Optional editable anemoi-core checkout:
#   sbatch --export=ALL,data=/path/to/aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr,anemoi_core_host=/path/to/anemoi-core correctness.slurm
# Optional benchmark overrides:
#   sbatch --export=ALL,data=/path/to/aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr,batch_limit=1,correctness_batch_limit=1,max_epochs=1 correctness.slurm

#SBATCH --job-name=correctness
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
container=${AIFS_IMAGE}
container_name=${container_name:-aifs-correctness}
benchmark_command=${benchmark_command:-correctness}
anemoi_core_host=${anemoi_core_host:-${ANEMOI_CORE_HOST:-}}
aifs_benchmark_host=${aifs_benchmark_host:-${AIFS_BENCHMARK_HOST:-$script_dir/docker/aifs-benchmark}}
data=${data:-}
default_correctness_dataset=aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr
correctness_dataset=$default_correctness_dataset
num_tasks=$((SLURM_NNODES * 4))
graph_dir_host="$script_dir/graphs"
mounts="$script_dir:/work,$graph_dir_host:/graphs"

if [[ -z "$data" ]]; then
  echo "data is required; pass the actual correctness .zarr directory" >&2
  exit 1
fi

if [[ ! -d "$data" ]]; then
  echo "data does not exist or is not a directory: $data" >&2
  exit 1
fi

if [[ "$data" != *.zarr ]]; then
  echo "data must point to the actual correctness .zarr directory: $data" >&2
  exit 1
fi

if [[ ! -f "$aifs_benchmark_host" ]]; then
  echo "aifs_benchmark_host does not exist or is not a file: $aifs_benchmark_host" >&2
  exit 1
fi

aifs_benchmark_host=$(cd "$(dirname "$aifs_benchmark_host")" && pwd -P)/$(basename "$aifs_benchmark_host")
mounts="$mounts,$aifs_benchmark_host:/opt/aifs-raps/bin/aifs-benchmark:ro"

data=$(cd "$data" && pwd -P)
mounts="$mounts,$data:/data/correctness/$correctness_dataset:ro"

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
export correctness_dataset

srun \
  -n "$num_tasks" \
  --mpi=pmix \
  --container-image="$container" \
  --container-mounts "$mounts" \
  --container-name "$container_name" \
  --container-workdir="/work" \
  aifs-benchmark "$benchmark_command"
