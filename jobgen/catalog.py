"""Resolve the four public benchmark profiles into jobs."""

from __future__ import annotations

from dataclasses import replace

from jobgen.cluster import ClusterConfig
from jobgen.inference import inference_jobs
from jobgen.models import Job, Options, canonicalize_job, override, real_data, resource_tag, run_id, tasks_per_node
from jobgen.training import correctness_args, highres_args, throughput_args, training_cases


def representative_jobs(options: Options, profile: str, cluster: ClusterConfig) -> list[Job]:
    if options.data_mode == "artificial" and profile == "correctness":
        raise SystemExit("correctness benchmark requires --data real")
    if options.ensembles is not None and profile != "highres":
        raise SystemExit("--ensembles is only valid for highres")
    if (options.mapper_chunks is not None or options.processor_chunks is not None) and profile not in {"highres", "throughput"}:
        raise SystemExit("--mapper-chunks and --processor-chunks are only valid for training profiles")
    if options.pin_memory is not None and profile not in {"highres", "throughput"}:
        raise SystemExit("--pin-memory and --no-pin-memory are only valid for highres and throughput")
    if options.hydra_overrides and profile not in {"highres", "throughput"}:
        raise SystemExit("--set is only valid for highres and throughput")
    if options.lead_time is not None and profile != "inference":
        raise SystemExit("--lead-time is only valid for inference")
    if options.inference_checkpoint and profile != "inference":
        raise SystemExit("--inference-checkpoint is only valid for inference")
    if options.inference_input and profile != "inference":
        raise SystemExit("--inference-input is only valid for inference")

    jobs: list[Job] = []
    for case in training_cases(profile):
        nodes = override(options.nodes, case.nodes)
        tasks = tasks_per_node(options)
        workers = override(options.num_workers, case.workers)
        batch_limit = override(options.batch_limit, case.batch_limit)
        max_epochs = override(options.max_epochs, case.max_epochs)
        tag = resource_tag(nodes, tasks, workers)
        if case.benchmark == "highres":
            ensembles = override(options.ensembles, case.ensembles)
            if options.ensembles is not None:
                tag = f"{tag}-ens{ensembles}"
            args = highres_args(
                nodes=nodes, ens_size=ensembles, local_bs=1,
                batch_limit=batch_limit, max_epochs=max_epochs, num_workers=workers,
                pin_memory=override(options.pin_memory, True),
                mapper_chunks=override(options.mapper_chunks, case.mapper_chunks),
                processor_chunks=override(options.processor_chunks, case.processor_chunks),
                ntasks_per_node=tasks, extra_args=options.hydra_overrides,
            )
            env = ("CLONED_DATASET=1", "ANEMOI_BASE_SEED=42")
        else:
            args = throughput_args(
                nodes=nodes, batch_limit=batch_limit, max_epochs=max_epochs,
                num_workers=workers, ntasks_per_node=tasks,
                mapper_chunks=override(options.mapper_chunks, case.mapper_chunks),
                processor_chunks=override(options.processor_chunks, case.processor_chunks),
                pin_memory=override(options.pin_memory, True),
                profile=case.profiling, extra_args=options.hydra_overrides,
            )
            env = ("CLONED_DATASET=1",)
        case_id = run_id(case.id_profile, case.benchmark, options.data_mode, tag)
        jobs.append(Job(
            case.id_profile, case_id, case.benchmark, "anemoi-training", args,
            nodes, tasks, cluster.cpus_per_task,
            override(options.time_limit, case.time_limit), cluster.container, case_id,
            data=real_data(options, cluster, case.benchmark), env=env,
        ))

    if profile == "inference":
        jobs.extend(inference_jobs(options, cluster))
    if profile == "correctness":
        nodes = override(options.nodes, 1)
        tasks = tasks_per_node(options)
        tag = resource_tag(nodes, tasks, 8)
        case_id = run_id("correctness", "correctness", "real", tag)
        jobs.append(Job(
            "correctness", case_id, "correctness", "anemoi-training",
            correctness_args(nodes=nodes, ntasks_per_node=tasks),
            nodes, tasks, cluster.cpus_per_task,
            override(options.time_limit, "01:00:00"), cluster.container, case_id,
            data=cluster.data("correctness"), env=("ANEMOI_BASE_SEED=42",),
        ))
    return jobs


def selected_jobs(profile: str, options: Options, cluster: ClusterConfig) -> list[Job]:
    jobs = representative_jobs(options, profile, cluster)
    if options.nsys:
        jobs = [replace(job, job_name=f"{job.job_name}-nsys") for job in jobs]
    return [canonicalize_job(job) for job in jobs]
