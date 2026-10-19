"""
Generate trajectories with a trained DiffTraj model (script version of
process/Traj_Generation.ipynb, so it can run in a batch job on the HPC).

By default it generates one trajectory per row of head_test.npy, IN ORDER, so
synthetic[i] is conditioned on head_test[i] and can be compared 1:1 with
real_trajectories_test.npy[i].

Normalisation statistics come from traj_stats.npy, which prepare_data.py computes
on the TRAIN trips only.

Usage
    python generate.py                                   # newest checkpoint, head_test.npy
    python generate.py --model "DiffTraj/<run>/models/<timestamp>/unet_2000.pt"
    python generate.py --head head.npy                   # generate from the train conditions instead
    python generate.py --limit 50 --out-dir test_gen     # quick test

Outputs (in the run's Files/<timestamp>/ folder, next to its config.py):
    synthetic_trajectories.npy   object array of (L, 2) [lat, lng]
    synthetic_index.npy          row of --head used for each trajectory (0..N-1)
    generation_info.json         model, head file, settings
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from utils.config import args as config_args
from utils.Traj_UNet import Guide_UNet
from utils.utils import p_xt, resample_trajectory


def newest_checkpoint():
    ckpts = glob.glob("DiffTraj/*/models/*/unet_*.pt")
    if not ckpts:
        raise SystemExit("No checkpoint found under DiffTraj/*/models/*/ - pass --model")
    return max(ckpts, key=os.path.getmtime)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="unet_*.pt checkpoint (default: newest one)")
    p.add_argument("--head", default="head_test.npy", help="conditions to generate from")
    p.add_argument("--stats", default="traj_stats.npy", help="train-only stats from prepare_data.py")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--timesteps", type=int, default=100, help="sampling steps (notebook: 100)")
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out-dir", default=None, help="default: the run's Files/<timestamp>/ folder")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0, help="only the first N rows (quick test)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    model_path = Path(args.model or newest_checkpoint())
    # .../<run>/models/<timestamp>/unet_N.pt  ->  .../<run>/Files/<timestamp>/
    run_dir, stamp = model_path.parents[2], model_path.parent.name
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "Files" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in config_args.items()})

    stats = np.load(args.stats, allow_pickle=True).item()
    mean, std = np.asarray(stats["mean"]), np.asarray(stats["std"])
    len_mean, len_std = float(stats["len_mean"]), float(stats["len_std"])

    head_all = torch.from_numpy(np.load(args.head, allow_pickle=True)).float()
    if args.limit:
        head_all = head_all[:args.limit]
    n = len(head_all)
    print(f"model : {model_path}")
    print(f"head  : {args.head} ({n} rows)")
    print(f"stats : {args.stats}  mean={mean.tolist()} len_mean={len_mean:.2f}")
    print(f"output: {out_dir}")

    unet = Guide_UNet(config).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()

    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).to(device)
    skip = n_steps // args.timesteps
    seq = list(range(0, n_steps, skip))
    seq_next = [-1] + seq[:-1]

    gen_traj = []
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        head = head_all[start:start + args.batch_size].to(device)
        b = len(head)
        lengths = (head[:, 3] * len_std + len_mean).int().clamp(min=2).cpu().numpy()
        x = torch.randn(b, 2, config.data.traj_length, device=device)
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((b,), i, device=device, dtype=torch.float)
            next_t = torch.full((b,), j, device=device, dtype=torch.float)
            with torch.no_grad():
                pred_noise = unet(x, t, head)
                x = p_xt(x, pred_noise, t, next_t, beta, args.eta)
        trajs = x.cpu().numpy()[:, :2, :]
        for k in range(b):
            gen_traj.append(resample_trajectory(trajs[k].T, int(lengths[k])) * std + mean)
        print(f"  {min(start + b, n)}/{n} generated ({time.time() - t0:.0f}s)", flush=True)

    out = np.empty(len(gen_traj), dtype=object)
    for k, tr in enumerate(gen_traj):
        out[k] = tr
    np.save(out_dir / "synthetic_trajectories.npy", out, allow_pickle=True)
    np.save(out_dir / "synthetic_index.npy", np.arange(n))  # synthetic[i] <- head[i]
    (out_dir / "generation_info.json").write_text(json.dumps({
        "model": str(model_path), "head": args.head, "stats": args.stats, "n": n,
        "timesteps": args.timesteps, "eta": args.eta, "seed": args.seed,
        "batch_size": args.batch_size, "seconds": round(time.time() - t0, 1),
    }, indent=2))
    Path("DiffTraj").mkdir(exist_ok=True)
    Path("DiffTraj/last_generation.txt").write_text(str(out_dir))   # read by hpc/train_nyc.sbatch
    print(f"Saved {n} trajectories -> {out_dir / 'synthetic_trajectories.npy'}")


if __name__ == "__main__":
    main()
