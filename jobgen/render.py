"""Render checked-in launcher skeletons without interpreting shell syntax."""

from __future__ import annotations

import shlex
from pathlib import Path

from jobgen.cluster import ClusterConfig
from jobgen.inference import inference_yaml
from jobgen.models import Job


TEMPLATE_DIR = Path(__file__).with_name("templates")


def render_template(name: str, **values: object) -> str:
    """Substitute explicit ``@@NAME@@`` markers in a launcher skeleton."""
    text = (TEMPLATE_DIR / name).read_text()
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", str(value))
    unresolved = [line for line in text.splitlines() if "@@" in line]
    if unresolved:
        raise ValueError(f"unresolved template marker in {name}: {unresolved[0]}")
    return text


REPO_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONTAINER_NAME = "aifs"


def write_args(path: Path, args: tuple[str, ...]) -> None:
    path.write_text("\n".join(args) + "\n")


def script_heredoc(filename: str, marker: str, body: str) -> str:
    return f"""cat > "$host_job_dir/{filename}" <<'{marker}'
{body.rstrip()}
{marker}
"""


def nsys_profile_command(
    job: Job,
    command: str,
    *,
    command_prefix: str = "",
    rank_variable: str = "RANK",
) -> str:
    """Wrap one GPU worker and give every rank a distinct Nsight report."""
    return f'''CMD=({command})
NSYS_BIN=${{NSYS_BIN:-/usr/local/cuda/bin/nsys}}
NSYS_DIR="$RUNDIR/nsys"
NSYS_OUT="$NSYS_DIR/{job.safe_id}-$(hostname -s)-rank${{{rank_variable}}}-gpu${{AIFS_PHYSICAL_LOCAL_RANK}}"
mkdir -p "$NSYS_DIR"
exec {command_prefix}"$NSYS_BIN" profile \\
  --force-overwrite=true \\
  --trace=cuda,nvtx \\
  --cuda-graph-trace=node \\
  --sample=none \\
  --cpuctxsw=none \\
  --cudabacktrace=none \\
  -o "$NSYS_OUT" \\
  -- "${{CMD[@]}}"'''


def write_slurm(
    path: Path,
    job: Job,
    cluster: ClusterConfig,
    args_rel: str,
    yaml_rel: str | None,
    enable_mmnvl: bool,
    enable_nsys: bool,
) -> None:
    mount_target = {
        "throughput": "/data/throughput/aifs-benchmarking-o800-10years-cloned.zarr",
        "highres": "/data/high-res/aifs-benchmark-4km-2years-synthetic.zarr",
        "correctness": "/data/correctness/aifs-ea-an-oper-0001-mars-o48-1979-2022-6h-v6.zarr",
    }
    mounts = '"$script_dir:/work,$script_dir/graphs:/graphs'
    if job.data:
        mounts += f",{job.data}:{mount_target[job.benchmark]}:ro"
    if job.input_data:
        mounts += f",{job.input_data}:/data/inference/input.zarr:ro"
    if job.checkpoint:
        mounts += f",{job.checkpoint}:/checkpoint/inference-last.ckpt:ro"
    mounts += '"'
    slurm_header = "\n".join(f"#SBATCH {directive}" for directive in cluster.slurm_directives)
    if job.command in ("anemoi-training", "anemoi-inference"):
        numa_nodes, gpus_per_numa_node, omp_num_threads = cluster.numa_settings(
            job.ntasks_per_node, job.cpus_per_task
        )
        if job.command == "anemoi-inference":
            parent_tasks_per_node = job.ntasks_per_node
            slurm_cpus_per_task = omp_num_threads
        else:
            parent_tasks_per_node = 1
            slurm_cpus_per_task = job.cpus_per_task
    else:
        numa_nodes = gpus_per_numa_node = omp_num_threads = 0
        parent_tasks_per_node = job.ntasks_per_node
        slurm_cpus_per_task = job.cpus_per_task
    srun_tasks = job.nodes * parent_tasks_per_node
    setup_script = """#!/bin/bash
set -euo pipefail

export AIFS_INSTALL_DIR=/opt/aifs-raps
export VIRTUAL_ENV=/opt/aifs-raps/venv
export ANEMOI_CONFIG_PATH=/opt/aifs-raps
export ANEMOI_CONFIG_OVERRIDE_PATH=/opt/aifs-raps/settings.toml
export HYDRA_FULL_ERROR=1
export MLFLOW_ALLOW_FILE_STORE=true
export HOME=/tmp/aifs-raps
export PATH="$VIRTUAL_ENV/bin:$AIFS_INSTALL_DIR/bin:$PATH"
mkdir -p "$HOME"
source "$VIRTUAL_ENV/bin/activate"
"""
    env_exports = "\n".join(f"export {item}" for item in job.env)
    if job.command == "anemoi-training":
        config_script = f"""#!/bin/bash
set -euo pipefail

source "$AIFS_CONTAINER_JOB_DIR/setup-env.sh"
mkdir -p "$RUNDIR"
mapfile -t aifs_args < "{args_rel}"
anemoi-training "${{aifs_args[@]}}" --cfg all > "$RUNDIR/cfg-all.yaml"
"""
        train_parent_script = f"""#!/bin/bash
set -euo pipefail

source "$AIFS_CONTAINER_JOB_DIR/setup-env.sh"
mkdir -p "$RUNDIR/logs"
export OMP_NUM_THREADS={omp_num_threads}
exec torchrun --no-python \\
  --nnodes={job.nodes} \\
  --nproc-per-node={job.ntasks_per_node} \\
  --node_rank="$SLURM_PROCID" \\
  --master-addr="$MASTER_ADDR" \\
  --master-port="$MASTER_PORT" \\
  "$AIFS_CONTAINER_JOB_DIR/train-worker.sh"
"""
        train_worker_script = f"""#!/bin/bash
set -euo pipefail

if (( LOCAL_RANK < 0 || LOCAL_RANK >= {job.ntasks_per_node} )); then
  echo "invalid LOCAL_RANK=$LOCAL_RANK" >&2
  exit 2
fi

export AIFS_PHYSICAL_LOCAL_RANK=$LOCAL_RANK
export numa_node=$((AIFS_PHYSICAL_LOCAL_RANK / {gpus_per_numa_node}))
export OMP_NUM_THREADS={omp_num_threads}
exec numactl --cpunodebind="$numa_node" --membind="$numa_node" \\
  "$AIFS_CONTAINER_JOB_DIR/train-worker-bound.sh"
"""
        if enable_nsys:
            train_command = nsys_profile_command(
                job,
                '"$VIRTUAL_ENV/bin/anemoi-training" "${aifs_args[@]}"',
            )
        else:
            train_command = 'exec "$VIRTUAL_ENV/bin/anemoi-training" "${aifs_args[@]}"'
        train_bound_worker_script = f"""#!/bin/bash
set -euo pipefail

source "$AIFS_CONTAINER_JOB_DIR/setup-env.sh"
echo "AIFS_NUMA_BIND host=$(hostname -s) rank=${{RANK:-unset}} local_rank=$LOCAL_RANK cuda_visible=${{CUDA_VISIBLE_DEVICES:-unset}} numa_node=$numa_node omp_num_threads=$OMP_NUM_THREADS" >&2
taskset -pc "$$" >&2
numactl --show >&2
mapfile -t aifs_args < "{args_rel}"
export CUDA_VISIBLE_DEVICES=$AIFS_PHYSICAL_LOCAL_RANK
export LOCAL_RANK=0
echo "AIFS_GPU_ISOLATION rank=$RANK physical_gpu=$AIFS_PHYSICAL_LOCAL_RANK visible=$CUDA_VISIBLE_DEVICES logical_local_rank=$LOCAL_RANK" >&2
{train_command}
"""
        helper_scripts = "\n".join(
            (
                script_heredoc("setup-env.sh", "AIFS_SETUP_ENV", setup_script),
                script_heredoc("config-preview.sh", "AIFS_CONFIG_PREVIEW", config_script),
                script_heredoc("train-parent.sh", "AIFS_TRAIN_PARENT", train_parent_script),
                script_heredoc("train-worker.sh", "AIFS_TRAIN_WORKER", train_worker_script),
                script_heredoc(
                    "train-worker-bound.sh",
                    "AIFS_TRAIN_WORKER_BOUND",
                    train_bound_worker_script,
                ),
            )
        )
        cfg_srun_args = f"""
cfg_srun_args=(
  --overlap
  --exact
  --nodes=1
  -n 1
  --ntasks-per-node=1
  --cpus-per-task={job.cpus_per_task}
  --mpi=pmix
  --container-image={shlex.quote(job.image)}
  --container-mounts "$mounts"
  --container-name={DEFAULT_CONTAINER_NAME}
  --container-workdir=/work
)
"""
        config_preview = 'srun "${cfg_srun_args[@]}" bash "$AIFS_CONTAINER_JOB_DIR/config-preview.sh"\n'
        run_script = "train-parent.sh"
    else:
        if yaml_rel is None:
            raise ValueError("inference jobs require a yaml config")
        no_output = dict(item.split("=", 1) for item in job.env if "=" in item).get("no_output") == "1"
        inference_config_setup = ""
        inference_config_path = yaml_rel
        if not no_output:
            inference_config_path = '"$RUNDIR/inf.yaml"'
            inference_config_setup = f'''# Each independent writer needs its own output and template files.
export RUNDIR="$RUNDIR/node-${{SLURM_NODEID:-0}}-task-$SLURM_LOCALID"
mkdir -p "$RUNDIR/logs"
sed 's|${{oc.env:RUNDIR}}|'"$RUNDIR"'|g' \\
  "{yaml_rel}" > "$RUNDIR/inf.yaml"
sed 's|$RAPS_AIFS_ROOT_DIR|/opt/aifs-raps|g' \\
  /opt/aifs-raps/etc/aifs/inf/grib-templates/templates.yaml > "$RUNDIR/grib_templates.yaml"
'''
        inference_parent_script = f"""#!/bin/bash
set -euo pipefail

source "$AIFS_CONTAINER_JOB_DIR/setup-env.sh"
if (( SLURM_LOCALID < 0 || SLURM_LOCALID >= {job.ntasks_per_node} )); then
  echo "invalid SLURM_LOCALID=$SLURM_LOCALID" >&2
  exit 2
fi

export HOME=/tmp/aifs-raps-${{SLURM_JOB_ID:-manual}}-${{SLURM_NODEID:-0}}-$SLURM_LOCALID
mkdir -p "$HOME"
export AIFS_PHYSICAL_LOCAL_RANK=$SLURM_LOCALID
export numa_node=$((AIFS_PHYSICAL_LOCAL_RANK / {gpus_per_numa_node}))
export OMP_NUM_THREADS={omp_num_threads}
{inference_config_setup.rstrip()}
exec numactl --cpunodebind="$numa_node" --membind="$numa_node" \\
  "$AIFS_CONTAINER_JOB_DIR/inference-worker-bound.sh"
"""
        inference_command = f"anemoi-inference run {inference_config_path}"
        if enable_nsys:
            inference_command = nsys_profile_command(
                job,
                inference_command,
                command_prefix="env -u SLURM_NTASKS -u SLURM_PROCID ",
                rank_variable="SLURM_PROCID",
            )
        else:
            inference_command = f"exec env -u SLURM_NTASKS -u SLURM_PROCID {inference_command}"
        inference_bound_worker_script = f"""#!/bin/bash
set -euo pipefail

source "$AIFS_CONTAINER_JOB_DIR/setup-env.sh"
echo "AIFS_NUMA_BIND host=$(hostname -s) slurm_rank=$SLURM_PROCID physical_gpu=$AIFS_PHYSICAL_LOCAL_RANK cuda_visible=${{CUDA_VISIBLE_DEVICES:-unset}} numa_node=$numa_node omp_num_threads=$OMP_NUM_THREADS" >&2
taskset -pc "$$" >&2
numactl --show >&2
export CUDA_VISIBLE_DEVICES=$AIFS_PHYSICAL_LOCAL_RANK
echo "AIFS_GPU_ISOLATION slurm_rank=$SLURM_PROCID physical_gpu=$AIFS_PHYSICAL_LOCAL_RANK visible=$CUDA_VISIBLE_DEVICES" >&2
# Anemoi checks Slurm before other launchers. Hide the global task count so
# every process follows the world_size: 1 configuration.
{inference_command}
"""
        helper_scripts = "\n".join(
            (
                script_heredoc("setup-env.sh", "AIFS_SETUP_ENV", setup_script),
                script_heredoc("inference-parent.sh", "AIFS_INFERENCE_PARENT", inference_parent_script),
                script_heredoc(
                    "inference-worker-bound.sh",
                    "AIFS_INFERENCE_WORKER_BOUND",
                    inference_bound_worker_script,
                ),
            )
        )
        cfg_srun_args = ""
        config_preview = ""
        run_script = "inference-parent.sh"

    script_dir_to_repo = "/".join(".." for _ in path.parent.relative_to(REPO_DIR).parts) or "."
    text = render_template(
        "parallel.slurm.tmpl",
        CASE_ID=job.case_id,
        JOB_NAME=job.job_name,
        NODES=job.nodes,
        SLURM_HEADER=slurm_header,
        PARENT_TASKS_PER_NODE=parent_tasks_per_node,
        CPUS_PER_TASK=slurm_cpus_per_task,
        TIME_LIMIT=job.time_limit,
        SCRIPT_DIR_TO_REPO=script_dir_to_repo,
        SAFE_ID=job.safe_id,
        MOUNTS=mounts,
        NCCL_MNNVL_ENABLE=1 if enable_mmnvl else 0,
        ENV_EXPORTS=env_exports,
        HELPER_SCRIPTS=helper_scripts,
        SRUN_TASKS=srun_tasks,
        IMAGE=shlex.quote(job.image),
        CONTAINER_NAME=DEFAULT_CONTAINER_NAME,
        CFG_SRUN_ARGS=cfg_srun_args,
        CONFIG_PREVIEW=config_preview,
        RUN_SCRIPT=run_script,
    )
    path.write_text(text)
    path.chmod(0o755)


def write_jobs(
    output: Path,
    jobs: list[Job],
    cluster: ClusterConfig,
    *,
    enable_mmnvl: bool = False,
    enable_nsys: bool = False,
    force: bool = False,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "configs").mkdir(exist_ok=True)
    (output / "jobs").mkdir(exist_ok=True)
    (output / "jobs" / "logs").mkdir(exist_ok=True)
    (REPO_DIR / "logs").mkdir(exist_ok=True)

    planned_paths: list[Path] = []
    for job in jobs:
        planned_paths.extend(
            (
                output / "configs" / f"{job.safe_id}.args",
                output / "jobs" / f"{job.safe_id}.slurm",
            )
        )
        if job.command == "anemoi-inference":
            planned_paths.append(output / "configs" / f"{job.safe_id}.inf.yaml")
    collisions = [path for path in planned_paths if path.exists()]
    if collisions and not force:
        listed = "\n  ".join(str(path.relative_to(output)) for path in collisions)
        raise SystemExit(
            "refusing to overwrite existing generated artifacts; use --force to replace only these files:\n"
            f"  {listed}"
        )

    manifest_header = "profile\tcase_id\tbenchmark\tnodes\ttasks_per_node\tjob_name\tscript\tconfig"
    existing_rows: list[str] = []
    manifest_path = output / "manifest.tsv"
    if manifest_path.is_file():
        lines = manifest_path.read_text().splitlines()
        if lines and lines[0] == manifest_header:
            existing_rows = lines[1:]
        elif lines and not force:
            raise SystemExit(
                "refusing to replace an existing manifest with an incompatible schema; "
                "use --force to replace it"
            )
    generated_rows: list[str] = []
    for job in jobs:
        args_path = output / "configs" / f"{job.safe_id}.args"
        slurm_path = output / "jobs" / f"{job.safe_id}.slurm"
        write_args(args_path, job.args)
        yaml_rel = None
        config_path = args_path
        if job.command == "anemoi-inference":
            yaml_path = output / "configs" / f"{job.safe_id}.inf.yaml"
            yaml_path.write_text(inference_yaml(job))
            yaml_rel = f"/work/{yaml_path.relative_to(REPO_DIR)}"
            config_path = yaml_path
        write_slurm(
            slurm_path,
            job,
            cluster,
            f"/work/{args_path.relative_to(REPO_DIR)}",
            yaml_rel,
            enable_mmnvl,
            enable_nsys,
        )
        generated_rows.append(
            "\t".join(
                (
                    job.profile,
                    job.case_id,
                    job.benchmark,
                    str(job.nodes),
                    str(job.ntasks_per_node),
                    job.job_name,
                    str(slurm_path.relative_to(REPO_DIR)),
                    str(config_path.relative_to(REPO_DIR)),
                )
            )
        )
    generated_case_ids = {job.case_id for job in jobs}
    preserved_rows = [row for row in existing_rows if row.split("\t", 2)[1] not in generated_case_ids]
    manifest_path.write_text("\n".join([manifest_header, *preserved_rows, *generated_rows]) + "\n")
