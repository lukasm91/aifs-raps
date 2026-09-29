# AIFS benchmark jobs

This repository publishes four benchmark profiles from the AIFS job generator:
`highres`, `throughput`, `inference`, and `correctness`. `clusters/` contains
site settings, `jobgen/` defines workloads and renders jobs, `generated/` holds
outputs from the generator, and `docker/` builds the container.

## Workloads

| Profile | Default allocation | Workload |
| --- | --- | --- |
| `highres` | 64 nodes, 4 GPUs per node | O2560, 2048 channels, 2 ensembles, 180 training batches, 2 epochs |
| `throughput` | 4 nodes, 4 GPUs per node | O800, 1024 channels, 900 training batches, 2 epochs |
| `inference` | 1 node, 4 GPUs | Four independent 15-day forecasts; generates with-output and no-output jobs |
| `correctness` | 1 node, 4 GPUs | O48, seed 42, 1000 training and 100 validation batches, 5 epochs |

These defaults and the training arguments are defined in
[`jobgen/training.py`](jobgen/training.py). `--data real` is the default; select
`--data artificial` for high-resolution or throughput runs using the synthetic
Zarr datasets bundled in the image. Correctness requires real O48 data. For
inference, `--data artificial` uses the bundled O2560 input; `--data real` uses
`inference.input` from the cluster file unless `--inference-input` overrides it.
Both inference modes require a compatible checkpoint.

## Build the image

From the repository root:

```bash
docker build -f docker/Dockerfile -t aifs-benchmark:latest docker
```

The Dockerfile follows the current container build in the source harness. It
installs the same Anemoi, PyTorch, FlashAttention, and FA4 patch stack, fetches
public AIFS configuration and static assets by sparse checkout, and generates
synthetic throughput and high-resolution datasets. The build arguments
`BASE_IMAGE`, `ANEMOI_BRANCH`, and `ASSET_REF` select the base image, Anemoi
revision, and asset revision. Convert or publish the image in the format your
cluster's container runtime uses, then set `container` in a copy of
`clusters/example.json`.

## Configure a cluster

```bash
cp clusters/example.json clusters/site.json
```

Set the Slurm directives, container path, CPU and NUMA topology, and data paths
in `site.json`. The example uses recognizable input filenames and dummy parent
paths. `cpus_per_task` is the CPU allocation for the one training parent per
node; `torchrun` divides those CPUs evenly among its GPU workers. Inference
uses one Slurm task per GPU with the corresponding CPU share. The checkpoint
can also be supplied with `--inference-checkpoint`.

## Generate jobs

Generate each profile independently, because the appropriate node count and
options differ by workload:

```bash
./generate_aifs_jobs.py --config site --profile highres
./generate_aifs_jobs.py --config site --profile throughput
./generate_aifs_jobs.py --config site --profile inference
./generate_aifs_jobs.py --config site --profile correctness
```

For a run using bundled synthetic training data:

```bash
./generate_aifs_jobs.py --config site --profile throughput --data artificial
```

The generator writes `.args` files, inference YAML, Slurm scripts, and a
manifest under `generated/`. It preserves existing cases; pass `--force` to
replace a selected case. Common overrides include `-N`, `--tasks-per-node`,
`--num-workers`, `--mapper-chunks`, `--processor-chunks`, `--ensembles` for
high-resolution training, `--batch-limit`, `--max-epochs`, `--pin-memory`,
`--no-pin-memory`, `--time`, `--mmnvl`, `--nsys`, and repeatable `--set` Hydra
arguments. Run `./generate_aifs_jobs.py --help` for the full CLI.

Submit a generated script with `sbatch`, for example:

```bash
sbatch generated/jobs/throughput-real-4n-4g-nw4.slurm
```

Training jobs first resolve their configuration inside the container with
`anemoi-training --cfg all`, saving `$RUNDIR/cfg-all.yaml`. They then start one
Slurm task per node and use `torchrun` to launch one process per GPU. The
launcher records NCCL diagnostics, GPU assignment, and NUMA binding.

## Expand source datasets

The `dataset_expansion` entries in the cluster file describe real source Zarr
directories and expanded destinations. Generate single-node copy jobs with:

```bash
./generate_dataset_expansion_jobs.py --config site --dataset throughput
./generate_dataset_expansion_jobs.py --config site --dataset highres
```

The jobs retain source metadata and copy source time chunks into the longer
datasets. A populated destination is protected unless the launcher was
generated with `--replace-dataset`. `--force` only replaces the launcher.

A container build and benchmark run need a Docker-capable build host and an
appropriate Slurm GPU cluster. Generated shell syntax can be checked with
`bash -n generated/jobs/*.slurm`.
