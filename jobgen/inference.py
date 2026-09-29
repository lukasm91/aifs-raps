"""Pure rendering helpers for inference-specific configuration."""

from __future__ import annotations

from pathlib import Path

from jobgen.cluster import ClusterConfig
from jobgen.models import Job, Options, override, resource_tag, run_id, tasks_per_node


def inference_yaml(job: Job) -> str:
    env_map = dict(item.split("=", 1) for item in job.env if "=" in item)
    no_output = env_map["no_output"] == "1"
    input_path = (
        "/data/inference/input.zarr"
        if job.input_data
        else "/data/high-res/aifs-benchmark-4km-2years-synthetic.zarr"
    )
    output = "parallel_output: False\nwriters_per_device: 0\noutput: none"
    if not no_output:
        output = """parallel_output: True
writers_per_device: 1
output:
  grib:
    path: ${oc.env:RUNDIR}/aifs_out.grib
    templates:
      - samples:
          index_path: ${oc.env:RUNDIR}/grib_templates.yaml"""
    return f"""runner: parallel
world_size: 1
{output}
checkpoint: /checkpoint/inference-last.ckpt
input:
  dataset:
    dataset: {input_path}
date: 20230905
lead_time: {env_map["lead_time"]}
"""


def inference_jobs(options: Options, cluster: ClusterConfig) -> list[Job]:
    checkpoint_value = options.inference_checkpoint or cluster.get("inference", "checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value:
        raise SystemExit("inference requires --inference-checkpoint or inference.checkpoint")
    checkpoint = str(Path(checkpoint_value).expanduser().resolve())
    input_value = options.inference_input or cluster.get("inference", "input")
    input_data = (
        str(Path(input_value).expanduser().resolve())
        if isinstance(input_value, str) and input_value and options.data_mode == "real"
        else ""
    )
    jobs = []
    inference_nodes = override(options.nodes, 1)
    inference_tasks = tasks_per_node(options)
    inference_tag = resource_tag(inference_nodes, inference_tasks)
    for label, no_output in (("with-output", "0"), ("no-output", "1")):
        inference_id = run_id("inference", "inference", options.data_mode, inference_tag, label)
        jobs.append(
            Job(
                "inference",
                inference_id,
                "inference",
                "anemoi-inference",
                ("run", "${RUNDIR}/inf.yaml"),
                inference_nodes,
                inference_tasks,
                cluster.cpus_per_task,
                override(options.time_limit, "01:00:00"),
                cluster.container,
                inference_id,
                input_data=input_data,
                checkpoint=checkpoint,
                env=(
                    f"lead_time={override(options.lead_time, 360)}",
                    "ANEMOI_INFERENCE_NUM_CHUNKS_MAPPER=32",
                    "ANEMOI_INFERENCE_NUM_CHUNKS_PROCESSOR=32",
                    "TRANSPOSE_ON_GPU=1",
                    f"no_output={no_output}",
                    "ANEMOI_BASE_SEED=42",
                    "PYTHONPATH=/work/runtime-overrides/inference${PYTHONPATH:+:$PYTHONPATH}",
                    "ANEMOI_DISABLE_RUNNER_CPROFILE=1",
                ),
            )
        )
    return jobs
