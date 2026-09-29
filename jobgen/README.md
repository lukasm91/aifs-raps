# Job generator layout

- `training.py` keeps the exact training arguments and resource defaults.
- `catalog.py` maps the four public profiles to jobs.
- `inference.py` constructs the inference jobs and their YAML.
- `cluster.py` reads cluster settings and validates CPU/NUMA topology.
- `models.py` contains the job and option records.
- `render.py` writes arguments, helper scripts, Slurm launchers, and the manifest.
- `generator.py` parses the CLI and orchestrates generation.
- `templates/` contains the shared Slurm skeleton.

Each training job runs a configuration preview before `torchrun`, using the same
argument file as the GPU workers. The preview is saved as `cfg-all.yaml` in the
run directory.
