"""Data passed between case resolution and output rendering."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from jobgen.cluster import ClusterConfig


@dataclass(frozen=True)
class Job:
    profile: str
    case_id: str
    benchmark: str
    command: str
    args: tuple[str, ...]
    nodes: int
    ntasks_per_node: int
    cpus_per_task: int
    time_limit: str
    image: str
    job_name: str
    data: str = ""
    input_data: str = ""
    checkpoint: str = ""
    env: tuple[str, ...] = field(default_factory=tuple)

    @property
    def safe_id(self) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in self.case_id)


@dataclass(frozen=True)
class Options:
    data_mode: str = "real"
    nodes: int | None = None
    tasks_per_node: int | None = None
    num_workers: int | None = None
    batch_limit: int | None = None
    max_epochs: int | None = None
    time_limit: str | None = None
    pin_memory: bool | None = None
    mapper_chunks: int | None = None
    processor_chunks: int | None = None
    ensembles: int | None = None
    lead_time: int | None = None
    inference_checkpoint: str | None = None
    inference_input: str | None = None
    nsys: bool = False
    hydra_overrides: tuple[str, ...] = ()


def override(value, default):
    return default if value is None else value


def resource_tag(nodes: int, tasks_per_node: int, num_workers: int | None = None) -> str:
    """A stable resource label used in case IDs, files, jobs, and run output."""
    tag = f"{nodes}n-{tasks_per_node}g"
    return f"{tag}-nw{num_workers}" if num_workers is not None else tag


def run_id(profile: str, benchmark: str, data: str, resources: str, *variants: str) -> str:
    """Render every generated run identity with the same ordered schema."""
    parts = [profile]
    if benchmark != profile:
        parts.append(benchmark)
    parts.extend((data, resources, *variants))
    return "-".join(part for part in parts if part)


def tasks_per_node(options: Options, default: int = 4) -> int:
    return override(options.tasks_per_node, default)


def real_data(options: Options, cluster: ClusterConfig, key: str) -> str:
    return "" if options.data_mode == "artificial" else cluster.data(key)


def canonicalize_job(job: Job) -> Job:
    """Make one flat ID the source of truth for generated job artifacts."""
    run_id_value = job.job_name
    env = (
        f"RUNDIR=/work/output/{run_id_value}-${{SLURM_JOB_ID}}",
        *(item for item in job.env if not item.startswith("RUNDIR=")),
    )
    return replace(job, case_id=run_id_value, env=env)
