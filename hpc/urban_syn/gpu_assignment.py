"""Verify the one-GPU Slurm allocation and record its physical identity.

Run inside Apptainer before starting the experiment. CUDA device 0 may map
onto a different physical GPU on the host, so record both identities.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def collect_gpu_assignment() -> dict[str, str | int | None]:
    """Return CUDA and physical GPU identifiers for the running process.

    NVML may be unable to identify the host GPU because of cluster permissions.
    In that case the physical index/UUID remain null rather than guessed.
    """
    import torch

    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("SLURM_JOB_ID was not propagated into Apptainer.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA GPU.")

    torch.empty(1, device="cuda:0")  # Initialize CUDA for the NVML process lookup.
    torch.cuda.synchronize()
    result: dict[str, str | int | None] = {
        "gpu.name": torch.cuda.get_device_name(0),
        "gpu.cuda_device": "cuda:0",
        "gpu.physical_index": None,
        "gpu.uuid": None,
        "slurm.job_id": os.environ["SLURM_JOB_ID"],
        "slurm.node": os.getenv("SLURMD_NODENAME") or os.getenv("HOSTNAME"),
        "slurm.cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")
    }

    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as error:
        print(f"WARNING: NVML unavailable; physical GPU ID not recorded ({error}).")
        return result

    try:
        # Identify the GPU running *this process*, not an arbitrary GPU 0.
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                continue
            if any(item.pid == os.getpid() for item in processes):
                result["gpu.physical_index"] = index
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                result["gpu.uuid"] = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
                break
    except pynvml.NVMLError as error:
        print(f"WARNING: NVML GPU identification failed ({error}).")
    finally:
        pynvml.nvmlShutdown()
    return result


def main() -> None:
    """Write one JSON file per Slurm job and print its contents to the log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = collect_gpu_assignment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Assigned GPU:", json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
