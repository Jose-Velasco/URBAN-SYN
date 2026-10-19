# DiffTraj (NYC) environment. Only the software is in the image: docker-compose.yml mounts the
# URBAN-SYN folder at /workspace/URBAN-SYN (on the HPC, repo/ is bound with --bind), so editing code
# or rebuilding traj.npy / head.npy never needs a rebuild. For the HPC the image is converted to an
# Apptainer .sif, because HPC nodes have no internet.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# PyTorch with CUDA 11.8, the same build as the Diff-RNTraj image.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# numpy, pandas, matplotlib: main.py, generate.py, prepare_data.py
# pyarrow: prepare_data.py reads the parquet
# colored 1.x: utils/logger.py uses the 1.x API (fg, attr)
RUN pip install numpy pandas pyarrow matplotlib tqdm "colored==1.4.4"

RUN python -c "import torch, numpy, pandas, pyarrow, matplotlib, tqdm; from colored import fg, attr"

WORKDIR /workspace
CMD ["/bin/bash"]
