# FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Etc/UTC

# Even though we wont use ParHIP, KaHIP’s build system is still checking MPI.
# Installing OpenMPI is the simplest path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    cmake  \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    libgeos-dev \
    tzdata \
    openmpi-bin \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone KaHIP
RUN git clone https://github.com/KaHIP/KaHIP.git

WORKDIR /opt/KaHIP

# Build and install C++ binaries
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j"$(nproc)"

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     build-essential \
#     libglib2.0-0 \
#     libgl1 \
#     libgeos-dev \
#     libproj-dev \
#     proj-data \
#     proj-bin \
#     libgdal-dev \
#     gdal-bin \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/repo

COPY baselines/gan/ts_trajgen/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


CMD ["bash"]
# CMD ["/bin/bash"]
