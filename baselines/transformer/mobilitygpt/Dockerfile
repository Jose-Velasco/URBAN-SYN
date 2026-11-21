FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/mobilitygpt/repo

COPY repo/nyc_workflow/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /tmp/requirements.txt

COPY repo /workspace/mobilitygpt/repo
RUN python -m pip install --no-deps . \
    && python -c "import datasets, finetuners, mobilitygpt, nyc_workflow, preprocessing; print('MobilityGPT imports: ok')"

CMD ["bash"]
