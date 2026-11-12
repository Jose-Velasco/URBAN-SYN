"""DiffTraj: GPS trajectories, one per test move, generated from that move's conditions."""
from __future__ import annotations

import numpy as np

from evaluation.tables import Converted, gps_table

MODEL = "difftraj"
NETWORK = "tstrajgen"  # road tables come from FMM on TS-TrajGen's network, so road IDs are geo_ids


def convert(files: dict, options: dict) -> Converted:
    """Read DiffTraj's generated, real test and real training moves and return them in the common format.

    Each generated trajectory's ref_id is the ID of the test move it was conditioned on: synthetic_index.npy gives the
    move (synthetic[i] <- head_test[index[i]]) and traj_ids_test.npy its ID. DiffTraj has no time. The road tables are
    added afterwards from FMM matches of these GPS tables (fmm_prepare.py).
    """
    generated = np.load(files["generated"], allow_pickle=True)
    index = np.load(files["index"])
    real = np.load(files["real_test"], allow_pickle=True)
    real_ids = np.load(files["real_test_ids"], allow_pickle=True).astype(str)
    train = np.load(files["real_train"], allow_pickle=True)
    train_ids = np.load(files["real_train_ids"], allow_pickle=True).astype(str)

    if len(generated) != len(index):
        raise ValueError(f"{len(generated)} generated trajectories but {len(index)} index entries")
    if len(real) != len(real_ids):
        raise ValueError(f"{len(real)} real test moves but {len(real_ids)} IDs")
    if len(train) != len(train_ids):
        raise ValueError(f"{len(train)} real training moves but {len(train_ids)} IDs")

    gen_ids = [f"{MODEL}:{i}" for i in range(len(generated))]

    return Converted(
        model=MODEL,
        generated=gps_table(
            (gen_ids[i], "main", real_ids[int(index[i])], generated[i], None) for i in range(len(generated))
        ),
        real=gps_table((real_ids[i], "real_test", None, real[i], None) for i in range(len(real))),
        train=gps_table((train_ids[i], "real_train", None, train[i], None) for i in range(len(train))),
        unit="move",
        notes={"time": "none (DiffTraj generates coordinates only)"},
    )
