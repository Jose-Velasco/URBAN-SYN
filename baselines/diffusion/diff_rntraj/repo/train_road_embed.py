#!/usr/bin/env python3
"""
Train Diff-RNTraj's road_embed.txt with node2vec, using only NumPy + PyTorch.

An earlier version used PecanPy + gensim, which can't be installed on new
Python versions. This one has the same input and the same output file format.

How it works
------------
With p = q = 1 (the setting the earlier version used; the paper doesn't give
p or q), node2vec is:
  1. weighted random walks over the road graph (NumPy, all walks at once), then
  2. skip-gram with negative sampling on those walks (word2vec's model), here
     in PyTorch, so it runs on the GPU if there is one.

Input  (from build_graph_nyc.py):  data/NYC/graph/graph_edge.edgelist, graph_node_id2idx.txt
Output:                            data/NYC/graph/road_embed.txt
    N 64
    0 e_0 ... e_63
    1 e_0 ... e_63
    ...
Diff-RNTraj loads line i into SE[i + 1] (row 0 = padding). Roads with no
connection to any other road get a zero vector, like the earlier version.

Run (from baselines/diffusion/diff_rntraj/repo/):
    python train_road_embed.py                     # 5 epochs, paper walk settings
    python train_road_embed.py --epochs 1 --num-walks 10   # quick timing test
    python train_road_embed.py --epochs 1000       # the paper's iteration count

The embedding is saved after every epoch, so you can stop at any time
(Ctrl+C) and keep the last finished epoch.

Paper settings (Diff-RNTraj, Sec. 5): 100 walks per road, walk length 80,
window 10, dimension 64, 1000 iterations.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="node2vec (p=q=1) road embeddings with NumPy + PyTorch.")
    p.add_argument("--data-dir", default="./data/NYC", help="Diff-RNTraj dataset root (contains graph/).")
    p.add_argument("--dimensions", type=int, default=64, help="Embedding size (Diff-RNTraj: 64).")
    p.add_argument("--num-walks", type=int, default=100, help="Walks started from every road (paper: 100).")
    p.add_argument("--walk-length", type=int, default=80, help="Roads per walk (paper: 80).")
    p.add_argument("--window-size", type=int, default=10, help="Skip-gram context window (paper: 10).")
    p.add_argument("--epochs", type=int, default=5, help="Passes over all walks (paper: 1000).")
    p.add_argument("--negatives", type=int, default=5, help="Negative samples per pair (word2vec default: 5).")
    p.add_argument("--lr", type=float, default=0.01, help="SparseAdam learning rate.")
    p.add_argument("--batch-walks", type=int, default=256,
                   help="Walks per training step. Lower it if you run out of (GPU) memory.")
    p.add_argument("--keep-self-loops", action="store_true",
                   help="Let walks stay on the same road (the graph has a self-loop wherever a "
                        "trajectory stayed on a road). Off by default: those steps teach nothing.")
    p.add_argument("--device", default="auto", help="auto, cuda, cpu (or mps on a Mac).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-name", default="road_embed.txt", help="File name inside graph/.")
    return p.parse_args()


# Graph and random walks (NumPy)
def load_graph(graph_dir: Path, keep_self_loops: bool):
    node_map = graph_dir / "graph_node_id2idx.txt"
    edges = graph_dir / "graph_edge.edgelist"
    for f in (node_map, edges):
        if not f.exists():
            sys.exit(f"Missing {f}\nRun build_graph_nyc.py first.")

    idx = []
    with node_map.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                idx.append(int(line.split(",")[1]))
    n = len(idx)
    if sorted(idx) != list(range(n)):
        sys.exit("graph_node_id2idx.txt indices are not 0..N-1.")

    e = np.loadtxt(edges, ndmin=2)
    src, dst = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
    w = e[:, 2].astype(np.float64) if e.shape[1] > 2 else np.ones(len(e))
    n_self = int((src == dst).sum())
    if not keep_self_loops:
        keep = src != dst
        src, dst, w = src[keep], dst[keep], w[keep]

    order = np.lexsort((dst, src))
    src, dst, w = src[order], dst[order], w[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=n))])
    cumw = np.cumsum(w)
    rowsum = np.bincount(src, weights=w, minlength=n)
    base = np.concatenate([[0.0], cumw])[indptr[:-1]]          # cumulative weight before each row
    linked = (np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)) > 0
    return n, src, dst, indptr, cumw, rowsum, base, linked, n_self


def random_walks(n, dst, indptr, cumw, rowsum, base, num_walks, walk_length, rng) -> np.ndarray:
    """All walks at once: walks[i] starts at road i % n. -1 = walk ended at a dead end."""
    walks = np.full((n * num_walks, walk_length), -1, dtype=np.int32)
    cur = np.tile(np.arange(n, dtype=np.int64), num_walks)
    walks[:, 0] = cur
    for t in range(1, walk_length):
        alive = np.nonzero((cur >= 0) & (rowsum[np.maximum(cur, 0)] > 0))[0]
        c = cur[alive]
        target = base[c] + rng.random(len(c)) * rowsum[c]    # weighted choice of the next road
        pos = np.searchsorted(cumw, target, side="right")
        pos = np.clip(pos, indptr[c], indptr[c + 1] - 1)       # guard against float rounding
        nxt = np.full_like(cur, -1)
        nxt[alive] = dst[pos]
        walks[:, t] = nxt
        cur = nxt
    return walks


def save_embedding(path: Path, vectors: np.ndarray, linked: np.ndarray) -> None:
    n, dims = vectors.shape
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(f"{n} {dims}\n")
        zero = " ".join(["0"] * dims)
        for i in range(n):
            if linked[i]:
                out.write(f"{i} " + " ".join(f"{x:.9g}" for x in vectors[i]) + "\n")
            else:
                out.write(f"{i} {zero}\n")
    os.replace(tmp, path)


def fmt_time(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"


# Main
def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        sys.exit("PyTorch is not installed for this Python.\n"
                 f'Install it with:  "{sys.executable}" -m pip install torch\n'
                 "(or run this in the URBAN-SYN dev container, which already has it).")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    graph_dir = Path(args.data_dir) / "graph"
    out_path = graph_dir / args.output_name

    # graph + walks
    t0 = time.time()
    n, src, dst, indptr, cumw, rowsum, base, linked, n_self = load_graph(graph_dir, args.keep_self_loops)
    print(f"Roads: {n:,} | transitions: {len(src):,} "
          f"({'kept' if args.keep_self_loops else 'dropped'} {n_self:,} self-loops) | "
          f"roads with no link to another road: {(~linked).sum():,} (get zero vectors)")
    walks = random_walks(n, dst, indptr, cumw, rowsum, base, args.num_walks, args.walk_length, rng)
    tokens = int((walks >= 0).sum())
    print(f"Walks: {len(walks):,} x {args.walk_length} steps = {tokens:,} road visits "
          f"({walks.nbytes / 1e9:.2f} GB) in {fmt_time(time.time() - t0)}")

    # skip-gram with negative sampling
    counts = np.bincount(walks[walks >= 0], minlength=n).astype(np.float64)
    noise = torch.tensor(counts ** 0.75, dtype=torch.float, device=device)
    noise /= noise.sum()

    emb_in = torch.nn.Embedding(n, args.dimensions, sparse=True).to(device)
    emb_out = torch.nn.Embedding(n, args.dimensions, sparse=True).to(device)
    torch.nn.init.uniform_(emb_in.weight, -0.5 / args.dimensions, 0.5 / args.dimensions)
    torch.nn.init.zeros_(emb_out.weight)
    opt = torch.optim.SparseAdam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=args.lr)

    W = torch.from_numpy(walks)
    if device == "cuda" and walks.nbytes < 0.3 * torch.cuda.get_device_properties(0).total_memory:
        W = W.to(device)                                        # keep all walks on the GPU if they fit
    n_batches = math.ceil(len(W) / args.batch_walks)
    win, K = args.window_size, args.negatives
    print(f"Training on {device} | {args.epochs} epoch(s) x {n_batches:,} steps "
          f"| window {win}, {K} negatives, dim {args.dimensions}\n")

    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        perm = torch.randperm(len(W), device=W.device)
        total_loss, total_pairs = 0.0, 0
        for bi in range(n_batches):
            b = W[perm[bi * args.batch_walks:(bi + 1) * args.batch_walks]].to(device).long()
            reach = torch.randint(1, win + 1, b.shape, device=device)   # word2vec's random window
            centers, contexts = [], []
            for d in range(1, win + 1):
                left, right = b[:, :-d], b[:, d:]
                ok = (left >= 0) & (right >= 0)
                fwd = ok & (reach[:, :-d] >= d)                 # center left, context right
                bwd = ok & (reach[:, d:] >= d)                  # center right, context left
                centers += [left[fwd], right[bwd]]
                contexts += [right[fwd], left[bwd]]
            ctr, ctx = torch.cat(centers), torch.cat(contexts)
            if ctr.numel() == 0:
                continue
            neg = torch.multinomial(noise, ctr.numel() * K, replacement=True).view(-1, K)

            u = emb_in(ctr)                                     # P x D
            v = emb_out(ctx)                                    # P x D
            nv = emb_out(neg)                                   # P x K x D
            pos_score = F.logsigmoid((u * v).sum(-1))
            neg_score = F.logsigmoid(-torch.bmm(nv, u.unsqueeze(-1)).squeeze(-1)).sum(-1)
            loss = -(pos_score + neg_score).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * ctr.numel()
            total_pairs += ctr.numel()
            if bi % max(1, n_batches // 20) == 0 or bi == n_batches - 1:
                done = (epoch - 1) + (bi + 1) / n_batches
                eta = (time.time() - t_train) / done * (args.epochs - done)
                print(f"\r  epoch {epoch}/{args.epochs}  {100 * (bi + 1) / n_batches:5.1f}%  "
                      f"loss {total_loss / max(total_pairs, 1):.4f}  "
                      f"time left ~{fmt_time(eta)}   ", end="", flush=True)

        vectors = emb_in.weight.detach().float().cpu().numpy()
        save_embedding(out_path, vectors, linked)
        print(f"\r  epoch {epoch}/{args.epochs}  loss {total_loss / max(total_pairs, 1):.4f}  "
              f"{total_pairs / 1e6:,.0f}M pairs in {fmt_time(time.time() - t_ep)}  -> saved{' ' * 20}")

    # sanity check: roads that follow each other should end up similar
    vec = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9)
    real = np.nonzero(src != dst)[0]
    pick = rng.choice(real, size=min(20000, len(real)), replace=False)
    edge_cos = float(np.mean(np.sum(vec[src[pick]] * vec[dst[pick]], axis=1)))
    a, b = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    rand_cos = float(np.mean(np.sum(vec[a] * vec[b], axis=1)))
    print(f"\nSanity check (cosine similarity): connected roads {edge_cos:.3f} vs random pairs {rand_cos:.3f} "
          + ("- connected roads are much more similar" if edge_cos > rand_cos + 0.2
             else "- not much difference yet, train more epochs"))
    print(f"Done in {fmt_time(time.time() - t0)}. Embedding: {out_path}  (header: {n} {args.dimensions})")


if __name__ == "__main__":
    main()
