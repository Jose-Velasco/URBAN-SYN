# Diff-RNTraj (NYC): one image with everything this baseline needs, from the raw NYC data to generated trajectories. 
# Build it once (docker compose build); the data pipeline (fmm_scripts/diffrntraj_version/run_diffrntraj_pipeline.py), 
# training and generation all run in it. For the HPC it is converted to an Apptainer .sif because HPC nodes have no internet.
#
#   FMM (ubodt_gen, fmm)   built like the TS-TrajGen fmm_scripts/dockerfile (Ubuntu 22.04,
#                          same packages and build), at the FMM commit that dockerfile clones.
#   conda env "diffrntraj" the default `python`: multi_main.py, generate_data.py,
#                          train_road_embed.py (torch with CUDA 11.8).
#   conda env "prep"       `prep-python`: prepare_fmm_inputs_diffrntraj.py (osmnx),
#                          convert_fmm_to_diffrntraj.py, build_graph_nyc.py.
# Two envs because the model needs networkx < 3 (common/road_network.py uses
# nx.read_shp), while osmnx and the data scripts use current packages.
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

# FMM: commit 19ef34e (2024-04-22) is the current master that fmm_scripts/dockerfile clones.
# make -j2 instead of -j$(nproc) so the build doesn't run out of memory on a laptop.
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

# Model env (same packages as before). GDAL and rtree install cleanly from conda-forge
# (pip usually fails on GDAL); networkx < 3 for nx.read_shp.
# Prep env: what the data pipeline used in the URBAN-SYN dev container
# (osmnx 2.0.7 + geopandas, pyogrio, pyarrow, pyyaml, shapely, networkx).
RUN /opt/conda/bin/mamba create -y -n diffrntraj -c conda-forge \
        python=3.11 gdal rtree "networkx<3" pandas matplotlib tqdm scikit-learn \
    && /opt/conda/bin/mamba create -y -n prep -c conda-forge \
        python=3.11 geopandas pyogrio pyarrow pyyaml shapely networkx pandas numpy tqdm requests \
    && /opt/conda/bin/mamba clean -afy

ENV PATH=/opt/conda/envs/diffrntraj/bin:/opt/fmm/wrappers:/opt/conda/bin:$PATH \
    PROJ_DATA=/opt/conda/envs/diffrntraj/share/proj \
    GDAL_DATA=/opt/conda/envs/diffrntraj/share/gdal

# PyTorch with CUDA 11.8 (same version as the project's uv.lock), in the model env.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# osmnx (same version the pipeline used before) in the prep env, and `prep-python`
# to run that env with its own PROJ/GDAL data.
RUN /opt/conda/envs/prep/bin/pip install osmnx==2.0.7 \
    && printf '#!/bin/sh\n# Python of the "prep" conda env (data pipeline), with its own PROJ/GDAL data.\nexport PROJ_DATA=/opt/conda/envs/prep/share/proj GDAL_DATA=/opt/conda/envs/prep/share/gdal\nexec /opt/conda/envs/prep/bin/python "$@"\n' > /usr/local/bin/prep-python \
    && chmod +x /usr/local/bin/prep-python

# Fail the build here if anything is missing, instead of later in a job.
RUN python -c "import torch, osgeo, rtree, sklearn, networkx; assert networkx.__version__ < '3'" \
    && prep-python -c "import osmnx, geopandas, pyogrio, pyarrow, yaml, shapely, networkx, tqdm; print('osmnx', osmnx.__version__)" \
    && test -x /usr/local/bin/fmm && test -x /usr/local/bin/ubodt_gen && command -v prep-python

WORKDIR /workspace
CMD ["/bin/bash"]
