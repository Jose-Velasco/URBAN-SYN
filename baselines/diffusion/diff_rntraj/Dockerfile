# Diff-RNTraj (NYC): the model image. Training (multi_main.py) and generation
# (generate_data.py) run in it. The FMM pipeline that builds the model input in repo/data/NYC/
# (including the road embedding) has its own image (fmm_scripts/diffrntraj_version/Dockerfile).
# For the HPC this image is converted to an Apptainer .sif, because HPC nodes have no internet.
#
# Code and data are not in the image: docker-compose.yml mounts the URBAN-SYN folder at
# /workspace/URBAN-SYN (on the HPC the job binds repo/), so editing code never needs a rebuild.
FROM condaforge/miniforge3:25.3.0-3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# GDAL and rtree install cleanly from conda-forge (pip usually fails on GDAL).
# networkx < 3: common/road_network.py uses nx.read_shp, removed in networkx 3.0.
RUN mamba create -y -n diffrntraj -c conda-forge \
        python=3.11 gdal rtree "networkx<3" pandas matplotlib tqdm scikit-learn \
    && mamba clean -afy

ENV PATH=/opt/conda/envs/diffrntraj/bin:$PATH \
    PROJ_DATA=/opt/conda/envs/diffrntraj/share/proj \
    GDAL_DATA=/opt/conda/envs/diffrntraj/share/gdal

# PyTorch with CUDA 11.8 (same version as the project's uv.lock).
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Fail the build here if anything is missing, instead of later in a job.
RUN python -c "import torch, osgeo, rtree, sklearn, networkx; assert networkx.__version__ < '3'"

WORKDIR /workspace
CMD ["/bin/bash"]
