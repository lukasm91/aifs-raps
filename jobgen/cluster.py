"""Shared, deliberately small JSON helpers for cluster configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def load_json_object(path: Path, *, missing_message: str, object_message: str) -> dict[str, object]:
    """Load one configuration file while leaving caller-specific errors intact."""
    if not path.is_file():
        raise SystemExit(missing_message.format(path=path))
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise SystemExit(object_message.format(path=path))
    return value


@dataclass(frozen=True)
class ClusterConfig:
    """Validated access to the site-specific facts used by generated jobs."""

    path: Path
    values: dict[str, object]

    @classmethod
    def load(cls, name_or_path: str, repo_dir: Path) -> ClusterConfig:
        path = Path(name_or_path)
        if not path.suffix:
            path = repo_dir / "clusters" / f"{name_or_path}.json"
        elif not path.is_absolute():
            path = repo_dir / path
        values = load_json_object(
            path,
            missing_message="config file not found: {path}",
            object_message="config file must contain a JSON object: {path}",
        )
        return cls(path=path, values=values)

    def get(self, *keys: str, default: object = None) -> object:
        value: object = self.values
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def require_string(self, *keys: str) -> str:
        value = self.get(*keys)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"config must define {'.'.join(keys)}")
        return value

    @property
    def container(self) -> str:
        return self.require_string("container")

    @property
    def cpus_per_task(self) -> int:
        value = self.get("cpus_per_task")
        if not isinstance(value, int) or value < 1:
            raise SystemExit("config must define cpus_per_task as a positive integer")
        return value

    @property
    def slurm_directives(self) -> tuple[str, ...]:
        value = self.get("slurm")
        if not isinstance(value, list):
            raise SystemExit("config must define slurm as a list")
        return tuple(str(item) for item in value)

    def data(self, key: str) -> str:
        return self.require_string("data", key)

    def numa_settings(self, workers_per_node: int, parent_cpus: int) -> tuple[int, int, int]:
        value = self.get("torchrun")
        if not isinstance(value, dict):
            raise SystemExit("config must define torchrun for generated training jobs")
        numa_nodes = value.get("numa_nodes")
        gpus_per_numa_node = value.get("gpus_per_numa_node")
        if not all(isinstance(item, int) and item > 0 for item in (numa_nodes, gpus_per_numa_node)):
            raise SystemExit("torchrun NUMA settings must be positive integers")
        available_workers = numa_nodes * gpus_per_numa_node
        if workers_per_node > available_workers:
            raise SystemExit(
                f"--tasks-per-node={workers_per_node} exceeds the {available_workers} GPUs described by torchrun"
            )
        if workers_per_node != 1 and workers_per_node % numa_nodes:
            raise SystemExit(
                f"--tasks-per-node must be 1 or divisible by torchrun.numa_nodes ({numa_nodes})"
            )
        workers_per_numa_node = 1 if workers_per_node == 1 else workers_per_node // numa_nodes
        if workers_per_numa_node > gpus_per_numa_node:
            raise SystemExit("--tasks-per-node exceeds the GPU capacity of a NUMA domain")
        if parent_cpus % workers_per_node:
            raise SystemExit("cpus_per_task must divide evenly among torchrun workers")
        return numa_nodes, workers_per_numa_node, parent_cpus // workers_per_node
