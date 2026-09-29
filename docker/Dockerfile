# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pyg:26.01-py3
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG ANEMOI_BRANCH=aifs-raps_20251215
ARG ASSET_REPO_URL=https://github.com/ecmwf/aifs-raps.git
ARG ASSET_REF=main

ENV AIFS_INSTALL_DIR=/opt/aifs-raps \
    AIFS_PYTORCH_CONSTRAINTS=/opt/aifs-raps/pytorch-constraints.txt \
    PATH=/opt/aifs-raps/venv/bin:/opt/aifs-raps/bin:${PATH} \
    PYTHONUNBUFFERED=1

# System tools used by the generated launchers and build steps.
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bc \
        curl \
        gawk \
        gdb \
        git \
        numactl \
        patch \
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

# The optional PyG extension speeds up graph construction in Anemoi.  Select
# the binary wheel matching the PyTorch version supplied by the base image.
RUN TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')" && \
    python -m pip install --no-cache-dir --no-build-isolation \
      -c "$AIFS_PYTORCH_CONSTRAINTS" torch-cluster \
      -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html"

# The PyG base image supplies FlashAttention 2.  Add the separate FA3 and FA4
# implementations too.  FA3 must be compiled against NVIDIA's nightly Torch.
RUN tmpdir=$(mktemp -d) && \
    git clone --depth 1 --recurse-submodules https://github.com/Dao-AILab/flash-attention.git "$tmpdir/flash-attention" && \
    cd "$tmpdir/flash-attention/hopper" && \
    MAX_JOBS=32 python setup.py install && \
    rm -rf "$tmpdir"

# Keep FA4's complete CuTe DSL stack pinned.  FA4 b27 + CuTe DSL 4.6.2 is the
# tested GB200 combination; letting pip choose a newer prerelease can silently
# reintroduce a compiler regression.
RUN python -m pip install --no-cache-dir \
      -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "nvidia-cutlass-dsl[cu13]==4.6.2" \
      "apache-tvm-ffi==0.1.12" \
      "torch-c-dlpack-ext==0.1.5" \
      "quack-kernels==0.6.4" && \
    python -m pip install --no-cache-dir --no-deps "flash-attn-4==4.0.0b27"

# Install Anemoi packages in separate resolver transactions; the git packages
# may intentionally upgrade dependencies installed by the older pinned wheels.
RUN python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" anemoi-utils==0.4.28 && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" anemoi-datasets==0.5.35 && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "anemoi-models[migrations] @ git+https://github.com/ecmwf/anemoi-core.git@$ANEMOI_BRANCH#subdirectory=models" && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "anemoi-training[profile] @ git+https://github.com/ecmwf/anemoi-core.git@$ANEMOI_BRANCH#subdirectory=training" && \
    python -m pip install --no-cache-dir -c "$AIFS_PYTORCH_CONSTRAINTS" \
      "anemoi-inference @ git+https://github.com/ecmwf/anemoi-inference.git@$ANEMOI_BRANCH" && \
    python -c 'from importlib.metadata import version; print("torch==" + version("torch")); print("torchvision==" + version("torchvision"))' \
      | diff -u "$AIFS_PYTORCH_CONSTRAINTS" - && \
    anemoi-training --help >/dev/null && \
    anemoi-inference --help >/dev/null

# The selected Anemoi branch predates its FA4 selector/wrapper.  This only
# exposes FA4; it intentionally does not alter Q/K/V tensor shapes.
COPY anemoi-fa4-wrapper.patch /tmp/
RUN cd "$(python -c 'from pathlib import Path; import anemoi.models; print(Path(anemoi.models.__file__).resolve().parent / "layers")')" && \
    patch --batch --forward -p0 < /tmp/anemoi-fa4-wrapper.patch && \
    rm /tmp/anemoi-fa4-wrapper.patch

# Install AIFS-RAPS benchmark configuration and static assets.
RUN templates_dir="$AIFS_INSTALL_DIR/etc/aifs/inf/grib-templates" && \
    tmpdir=$(mktemp -d) && \
    repo_dir="$tmpdir/aifs-raps" && \
    git clone --depth 1 --filter=blob:none --sparse --branch "$ASSET_REF" \
      "$ASSET_REPO_URL" "$repo_dir" && \
    git -C "$repo_dir" sparse-checkout set --no-cone \
      '/etc/aifs/config/*' \
      '/etc/aifs/tarballs/grids.tar.gz' \
      '/etc/aifs/inf/grib-templates/*' && \
    rm -rf "$templates_dir" "$AIFS_INSTALL_DIR/config" "$AIFS_INSTALL_DIR/grids" && \
    mkdir -p "$templates_dir" "$AIFS_INSTALL_DIR/grids" && \
    cp -a "$repo_dir/etc/aifs/inf/grib-templates/." "$templates_dir/" && \
    tar -xzf "$repo_dir/etc/aifs/tarballs/grids.tar.gz" \
      -C "$AIFS_INSTALL_DIR/grids" \
      --strip-components=4 \
      home/mlx/ai-ml/grids && \
    cp -a "$(python -c 'from pathlib import Path; import anemoi.training; print(Path(anemoi.training.__file__).resolve().parent / "config")')" "$AIFS_INSTALL_DIR/config" && \
    cp -a "$repo_dir/etc/aifs/config/raps.yaml" "$AIFS_INSTALL_DIR/config/raps.yaml" && \
    cp -a "$repo_dir/etc/aifs/config/zarr_no_insolation.yaml" "$AIFS_INSTALL_DIR/config/data/zarr_no_insolation.yaml" && \
    cp -a "$repo_dir/etc/aifs/config/less_memory_loss_ensemble.yaml" "$AIFS_INSTALL_DIR/config/training/less_memory_loss_ensemble.yaml" && \
    printf '[utils]\ngrids_path = "%s/grids"\n' "$AIFS_INSTALL_DIR" > "$AIFS_INSTALL_DIR/settings.toml" && \
    rm -rf "$tmpdir"

# Runtime dataset generator.
COPY --chmod=0755 make_synthetic_benchmark_zarr.py /opt/aifs-raps/bin/

# Generate compact dummy datasets. Their data chunks are hardlinks, so the
# logical Zarr size is much larger than the image layer size.
RUN mkdir -p /data/throughput /graphs /work && \
    python /opt/aifs-raps/bin/make_synthetic_benchmark_zarr.py \
      --throughput /data/throughput/aifs-benchmarking-o800-10years-cloned.zarr

RUN mkdir -p /data/high-res && \
    python /opt/aifs-raps/bin/make_synthetic_benchmark_zarr.py \
      --highres /data/high-res/aifs-benchmark-4km-2years-synthetic.zarr

WORKDIR /work
CMD ["bash"]
