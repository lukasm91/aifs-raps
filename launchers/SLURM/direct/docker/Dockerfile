# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pyg:26.01-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ANEMOI_BRANCH=aifs-raps_20251215
ARG ASSET_BASE_URL=https://raw.githubusercontent.com/ecmwf/aifs-raps/main/etc/aifs

ENV AIFS_INSTALL_DIR=/opt/aifs-raps \
    AIFS_PYTORCH_CONSTRAINTS=/opt/aifs-raps/pytorch-constraints.txt \
    PATH=/opt/aifs-raps/venv/bin:/opt/aifs-raps/bin:${PATH} \
    PYTHONUNBUFFERED=1

# System tools used by the benchmark launcher and build steps.
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bc \
        curl \
        gawk \
        gdb \
        git \
        numactl \
        python3-dbg \
        python3-venv \
        rsync \
        wget && \
    rm -rf /var/lib/apt/lists/*

# Isolate AIFS-RAPS Python packages from the NVIDIA base image Python.
RUN python3 -m venv /opt/aifs-raps/venv --system-site-packages && \
    python -m pip install --upgrade pip setuptools wheel

# Keep the PyTorch stack from the NVIDIA base image. Anemoi dependency
# resolution is constrained to these exact installed versions so pip cannot
# shadow them with wheels from PyPI or the PyTorch index.
RUN python -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("torchvision==" + version("torchvision"))' \
      > "$AIFS_PYTORCH_CONSTRAINTS"

# Install Anemoi packages in separate resolver transactions; the git packages
# may intentionally upgrade dependencies installed by the older pinned wheels.
RUN python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" anemoi-utils==0.4.28 && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" anemoi-datasets==0.5.35 && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "anemoi-models[migrations] @ git+https://github.com/ecmwf/anemoi-core.git@$ANEMOI_BRANCH#subdirectory=models" && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "anemoi-training[profile] @ git+https://github.com/ecmwf/anemoi-core.git@$ANEMOI_BRANCH#subdirectory=training" && \
    python -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("torchvision==" + version("torchvision"))' \
      | diff -u "$AIFS_PYTORCH_CONSTRAINTS" - && \
    anemoi-training --help >/dev/null

# Download and wire benchmark configuration into Anemoi's config tree.
RUN asset_dir="$AIFS_INSTALL_DIR/assets" && \
    mkdir -p "$asset_dir/config" && \
    curl -fsSL "$ASSET_BASE_URL/config/raps.yaml" \
      -o "$asset_dir/config/raps.yaml" && \
    curl -fsSL "$ASSET_BASE_URL/config/zarr_no_insolation.yaml" \
      -o "$asset_dir/config/zarr_no_insolation.yaml" && \
    curl -fsSL "$ASSET_BASE_URL/config/less_memory_loss_ensemble.yaml" \
      -o "$asset_dir/config/less_memory_loss_ensemble.yaml" && \
    source /opt/aifs-raps/venv/bin/activate && \
    rm -rf "$AIFS_INSTALL_DIR/config" && \
    cp -a "$(python -c 'from pathlib import Path; import anemoi.training; print(Path(anemoi.training.__file__).resolve().parent / "config")')" "$AIFS_INSTALL_DIR/config" && \
    ln -sfn "$asset_dir/config/raps.yaml" "$AIFS_INSTALL_DIR/config/raps.yaml" && \
    ln -sfn "$asset_dir/config/zarr_no_insolation.yaml" "$AIFS_INSTALL_DIR/config/data/zarr_no_insolation.yaml" && \
    ln -sfn "$asset_dir/config/less_memory_loss_ensemble.yaml" "$AIFS_INSTALL_DIR/config/training/less_memory_loss_ensemble.yaml"

# Install the prebuilt grid assets used by the Anemoi configs.
RUN asset_dir="$AIFS_INSTALL_DIR/assets" && \
    mkdir -p "$asset_dir/tarballs" && \
    curl -fsSL "$ASSET_BASE_URL/tarballs/grids.tar.gz" \
      -o "$asset_dir/tarballs/grids.tar.gz" && \
    tmpdir=$(mktemp -d) && \
    tar -xzf "$asset_dir/tarballs/grids.tar.gz" -C "$tmpdir" && \
    rm -rf "$AIFS_INSTALL_DIR/grids" && \
    mv "$tmpdir/home/mlx/ai-ml/grids" "$AIFS_INSTALL_DIR/grids" && \
    rm -rf "$tmpdir"

# Point Anemoi at the image-owned grid directory.
RUN printf '[utils]\ngrids_path = "%s/grids"\n' "$AIFS_INSTALL_DIR" > "$AIFS_INSTALL_DIR/settings.toml"

# Runtime tools and the dataset generator.
COPY --chmod=0755 make_random_hardlink_zarr.py aifs-benchmark /opt/aifs-raps/bin/

# Generate compact dummy datasets. Their data chunks are hardlinks, so the
# logical Zarr size is much larger than the image layer size.
RUN mkdir -p /data/throughput /graphs /work && \
    python /opt/aifs-raps/bin/make_random_hardlink_zarr.py \
      --throughput /data/throughput/aifs-benchmarking-o800-10years-cloned.zarr

RUN mkdir -p /data/high-res && \
    python /opt/aifs-raps/bin/make_random_hardlink_zarr.py \
      --highres /data/high-res/aifs-benchmark-4km-2years-synthetic.zarr

WORKDIR /work
CMD ["aifs-benchmark", "help"]
