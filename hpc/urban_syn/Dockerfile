# Runtime environment for joint GNN + Diff-RNTraj experiments.
# Build from the URBAN-SYN repository root:
#   docker build -f hpc/joint_diff_rntraj/Dockerfile -t urban-syn-joint:cu118 .
#
# This image installs the locked Python dependencies only. The repository,
# processed graph, input trajectories, and experiment outputs are mounted at
# /workspace at execution time; none are baked into the image.
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/workspace/src

# Runtime libraries for geospatial packages, scientific Python, and PyTorch.
# CUDA runtime dependencies come from the locked PyTorch cu118 wheels; the
# NVIDIA driver is supplied by Docker's GPU runtime or Apptainer --nv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Pin uv rather than installing the mutable `latest` image.
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /usr/local/bin/

# A small, stable build context keeps dependency layers reusable when source
# files or experiment YAMLs change. The project itself is not installed here.
WORKDIR /opt/urban-syn-build
COPY pyproject.toml uv.lock ./

# --locked fails if uv.lock no longer matches pyproject.toml.
# --no-install-project installs third-party dependencies without requiring
# the URBAN-SYN source tree to be present at build time.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --python /usr/local/bin/python3.11

# Fail the build early if core scientific and experiment dependencies cannot
# import. GPU availability is checked separately at execution time.
RUN python -c "import torch, torch_geometric, geopandas, mlflow; print('PyTorch:', torch.__version__, '| CUDA wheels:', torch.version.cuda)"

WORKDIR /workspace
CMD ["bash"]
