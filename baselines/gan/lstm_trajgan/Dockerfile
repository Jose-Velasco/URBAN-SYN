FROM tensorflow/tensorflow:1.13.1-gpu-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/repo

# Keep pip tooling conservative for this old stack
RUN pip install --no-cache-dir --upgrade "pip<22" "setuptools<60" "wheel"

# pandas and scikit-learn have issues being installed by pip so insatlling them with no
# exact version
RUN pip install --no-cache-dir \
    pandas \
    Keras==2.2.4 \
    geohash2==1.1 \
    scikit-learn

# Below are requirements listed by orginal repo
# RUN pip install --no-cache-dir \
#     numpy==1.18.4 \
#     pandas==1.1.5 \
#     Keras==2.2.4 \
#     geohash2==1.1 \
#     scikit-learn==0.23.2

CMD ["/bin/bash"]
