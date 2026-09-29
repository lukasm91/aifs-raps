#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from jobgen.catalog import selected_jobs
from jobgen.cluster import ClusterConfig
from jobgen.models import Options
from jobgen.render import write_jobs


REPO_DIR = Path(__file__).resolve().parent.parent


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AIFS benchmark configs and submit scripts.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=("highres", "throughput", "inference", "correctness"),
        required=True,
        help="The single benchmark profile to generate.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Cluster config name under clusters/<name>.json, or a JSON file path.",
    )
    parser.add_argument("--output", type=Path, default=REPO_DIR / "generated")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the generated artifacts selected by this invocation.",
    )
    parser.add_argument(
        "--data",
        choices=("real", "artificial"),
        default="real",
        help="Dataset source for training and inference. Correctness requires real data.",
    )
    parser.add_argument("-N", "--nodes", type=positive_int, help="Override Slurm nodes and matching training config.")
    parser.add_argument(
        "--tasks-per-node",
        type=positive_int,
        help="Override GPU workers per node for training jobs (including correctness).",
    )
    parser.add_argument("--num-workers", type=positive_int, help="Override training dataloader workers.")
    parser.add_argument(
        "--mapper-chunks",
        type=positive_int,
        help="Override training encoder and decoder chunk counts.",
    )
    parser.add_argument(
        "--processor-chunks",
        type=positive_int,
        help="Override training processor chunk count.",
    )
    parser.add_argument(
        "--ensembles",
        type=positive_int,
        help="Override the number of ensembles for a high-resolution profile.",
    )
    parser.add_argument("--batch-limit", type=positive_int, help="Override training batch limit.")
    parser.add_argument("--max-epochs", type=positive_int, help="Override training max epochs.")
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="Enable DataLoader pinned memory (overrides the profile default).",
    )
    pin_memory_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="Disable DataLoader pinned memory (overrides the profile default).",
    )
    parser.set_defaults(pin_memory=None)
    parser.add_argument("--time", dest="time_limit", help="Override Slurm time limit for generated jobs.")
    parser.add_argument("--lead-time", type=positive_int, help="Override inference lead time.")
    parser.add_argument(
        "--inference-checkpoint",
        help="Checkpoint file consumed by inference (for example, .../inference-last.ckpt).",
    )
    parser.add_argument(
        "--inference-input",
        help=(
            "External Zarr input for inference. By default, use the bundled "
            "/data/high-res/aifs-benchmark-4km-2years-synthetic.zarr dataset."
        ),
    )
    parser.add_argument(
        "--mmnvl",
        action="store_true",
        help="Enable multi-node NVLink for generated jobs (default: NCCL_MNNVL_ENABLE=0).",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Profile training or inference GPU work with Nsight Systems under RUNDIR/nsys.",
    )
    parser.add_argument(
        "--set",
        dest="hydra_overrides",
        action="append",
        default=[],
        metavar="HYDRA_ARG",
        help="Append a raw Hydra override to generated training .args files. Repeat for multiple overrides.",
    )
    args = parser.parse_args()

    if len(args.profile) != 1:
        raise SystemExit("select exactly one --profile per invocation")

    cluster = ClusterConfig.load(args.config, REPO_DIR)
    options = Options(
        data_mode=args.data,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        num_workers=args.num_workers,
        batch_limit=args.batch_limit,
        max_epochs=args.max_epochs,
        time_limit=args.time_limit,
        pin_memory=args.pin_memory,
        mapper_chunks=args.mapper_chunks,
        processor_chunks=args.processor_chunks,
        ensembles=args.ensembles,
        lead_time=args.lead_time,
        inference_checkpoint=args.inference_checkpoint,
        inference_input=args.inference_input,
        nsys=args.nsys,
        hydra_overrides=tuple(args.hydra_overrides),
    )
    output = args.output.resolve()
    try:
        output.relative_to(REPO_DIR)
    except ValueError as exc:
        raise SystemExit(f"--output must be inside {REPO_DIR}") from exc
    jobs = selected_jobs(args.profile[0], options, cluster)
    write_jobs(
        output, jobs, cluster, enable_mmnvl=args.mmnvl, enable_nsys=args.nsys, force=args.force,
    )
    print(f"wrote {len(jobs)} jobs to {output}")


if __name__ == "__main__":
    main()
