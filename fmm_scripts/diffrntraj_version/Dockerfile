# Diff-RNTraj FMM pipeline (NYC): the image that turns the NYC data into everything training
# needs (run_diffrntraj_pipeline.py in this folder: prep, ubodt, fmm, convert, graph, embed).
# Training and generation run in the model image (baselines/diffusion/diff_rntraj/Dockerfile),
# usually on the HPC.
#
#   FMM (ubodt_gen, fmm)   built like the TS-TrajGen fmm_scripts/dockerfile (Ubuntu 22.04,
#                          same packages and build), at the FMM commit that dockerfile clones.
#   Python (conda env)     the default `python`: osmnx 2.0.7, geopandas, pyogrio, pyarrow,
#                          pyyaml, shapely, networkx, pandas, numpy, tqdm, and torch 2.7.1
#                          with CUDA 11.8 for the road embedding (train_road_embed.py), which
#                          uses the GPU when there is one and the CPU otherwise.
#
# Code and data are not in the image: docker-compose.yml mounts the URBAN-SYN folder at
# /workspace/URBAN-SYN, so editing code never needs a rebuild.

FROM condaforge/miniforge3:25.3.0-3 AS conda

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# FMM build dependencies: the same system packages as the TS-TrajGen fmm_scripts/dockerfile.
RUN apt-get update && apt-get install -y \
        build-essential cmake git swig pkg-config curl wget ca-certificates \
        software-properties-common libboost-dev libboost-serialization-dev \
        libgdal-dev gdal-bin libbz2-dev libexpat1-dev \
        python3 python3-dev python3-pip python3-venv python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

ARG FMM_COMMIT=19ef34e1f57ff2f2484231aa0d01dfffea986ec1
RUN git clone https://github.com/cyang-kth/fmm.git /opt/fmm \
    && cd /opt/fmm && git checkout ${FMM_COMMIT} \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j2 \
    && make install \
    && ldconfig

# FMM uses Ubuntu's GDAL/PROJ, while PROJ_DATA/GDAL_DATA below point to the conda env's
# (newer) data. These wrappers (first on PATH) clear the two variables before running FMM.
RUN mkdir -p /opt/fmm/wrappers \
    && for b in fmm ubodt_gen stmatch; do \
         printf '#!/bin/sh\nunset PROJ_DATA GDAL_DATA\nexec /usr/local/bin/%s "$@"\n' "$b" > /opt/fmm/wrappers/$b; \
         chmod +x /opt/fmm/wrappers/$b; \
       done

COPY --from=conda /opt/conda /opt/conda

# Python env for the pipeline scripts (what the data pipeline used in the URBAN-SYN dev container).
RUN /opt/conda/bin/mamba create -y -n fmm -c conda-forge \
        python=3.11 geopandas pyogrio pyarrow pyyaml shapely networkx pandas numpy tqdm requests \
    && /opt/conda/bin/mamba clean -afy \
    && /opt/conda/envs/fmm/bin/pip install osmnx==2.0.7

ENV PATH=/opt/fmm/wrappers:/opt/conda/envs/fmm/bin:/opt/conda/bin:$PATH \
    PROJ_DATA=/opt/conda/envs/fmm/share/proj \
    GDAL_DATA=/opt/conda/envs/fmm/share/gdal

# PyTorch with CUDA 11.8 (same version as the model image and the project's uv.lock), for the
# road embedding.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Fail the build here if anything is missing, instead of later in the pipeline.
RUN python -c "import osmnx, geopandas, pyogrio, pyarrow, yaml, shapely, networkx, tqdm, torch; print('osmnx', osmnx.__version__, 'torch', torch.__version__)" \
    && test -x /usr/local/bin/fmm && test -x /usr/local/bin/ubodt_gen

WORKDIR /workspace/URBAN-SYN/fmm_scripts/diffrntraj_version
CMD ["/bin/bash"]
