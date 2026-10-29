"""Shared paired-evaluation protocol; no historical experiment paths or launchers."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon

CHECKPOINTS = ("model_best.pt", "model_pretrain.pt")
SEEDS = (42, 43, 44)
PROMPTS = (1, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def stable_seed(base_seed, trajectory_id, prompt_length):
    payload = json.dumps([int(base_seed), str(trajectory_id), int(prompt_length)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


def route_distance(roads, lengths):
    """Meters per occurrence, including revisits; never deduplicate roads."""
    return math.fsum(lengths[road] for road in roads)


def select_validation(rows, split, count=500, seed=42):
    train, validation = set(split["train_indices"]), set(split["validation_indices"])
    require(len(validation) == len(split["validation_indices"]), "Duplicate validation indices")
    require(not train & validation, "Training and validation indices overlap")
    require(train | validation == set(range(len(rows))), "Saved split does not partition all trajectory rows")
    train_parents = {rows[i]["parent_traj_id"] for i in train}
    val_parents = {rows[i]["parent_traj_id"] for i in validation}
    require(not train_parents & val_parents, "Parent leakage across saved split")
    require(train_parents == set(split["train_parents"]) and val_parents == set(split["validation_parents"]),
            "Saved parent IDs differ from trajectory CSV")
    require(len(validation) >= count, "Not enough distinct validation trajectories")
    ids = [row["traj_id"] for row in rows]
    require(len(set(ids)) == len(ids), "Trajectory IDs must be unique")
    indices = np.random.RandomState(seed).choice(sorted(validation), size=count, replace=False).tolist()
    selected = []
    for rank, index in enumerate(indices):
        row = rows[index]
        roads = list(map(int, row["rid_list"].split(",")))
        require(row["parent_traj_id"] in val_parents and row["parent_traj_id"] not in train_parents,
                "Selected parent outside validation partition")
        selected.append({"selection_rank": rank, "trajectory_index": index,
                         "trajectory_id": row["traj_id"], "parent_id": row["parent_traj_id"],
                         "reference_roads": roads,
                         "eligible_prompt_lengths": [n for n in PROMPTS if len(roads) > n]})
    return selected


@torch.inference_mode()
def generate_route(model, prompt_tokens, eos, seed, max_tokens=81):
    """Mirror generate_test at T=1, sampling, no top-k, exposing termination.

    Like the existing implementation, sample before the length-cap check and
    give EOS priority when both conditions coincide. The leading EOS counts
    toward the 81-token total; returned road routes therefore have at most 80 roads.
    """
    torch.manual_seed(seed)
    sequence = prompt_tokens.clone()
    steps, maximum_error = 0, 0.0
    while True:
        logits, _ = model(sequence[:, -model.block_size:])
        probabilities = torch.softmax(logits[:, -1, :], dim=-1)
        finite = torch.isfinite(probabilities).all().item()
        error = abs(probabilities.sum().item() - 1)
        require(finite and error < 1e-5, "Nonfinite or unnormalized generation probabilities")
        forbidden = model.adj_matrix[sequence[0, -1]].eq(0)
        require(torch.all(probabilities[0, forbidden] == 0).item(), "Forbidden probability is nonzero")
        maximum_error = max(maximum_error, error)
        steps += 1
        token = torch.multinomial(probabilities, 1)
        if token.item() == eos:
            reason = "eos"
            break
        if sequence.shape[1] == max_tokens:
            reason = "length_cap"
            break
        sequence = torch.cat((sequence, token), dim=1)
    return sequence[0, 1:].tolist(), reason, steps, maximum_error


def route_quality(roads, lengths, edges):
    valid = [road in lengths for road in roads]
    pairs = list(zip(roads, roads[1:]))
    bad = sum(pair not in edges for pair in pairs)
    return {"road_count": len(roads),
            "distance_m": route_distance(roads, lengths) if all(valid) else None,
            "valid_road_tokens": sum(valid), "valid_road_ids": all(valid),
            "adjacent_pairs": len(pairs), "invalid_transitions": bad,
            "has_invalid_transition": bool(bad), "single_road": len(roads) == 1,
            "has_repeated_road": len(set(roads)) < len(roads),
            "repeated_occurrences": len(roads) - len(set(roads)),
            "unique_road_fraction": len(set(roads)) / len(roads)}


def describe(values, bins=None):
    values = np.asarray(values, dtype=np.float64)
    result = {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
              "std": float(values.std()), "min": float(values.min()), "max": float(values.max()),
              "percentiles": {str(q): float(np.percentile(values, q)) for q in (5, 10, 25, 50, 75, 90, 95, 99)}}
    if bins is not None:
        histogram = np.histogram(values, bins=bins)[0]
        require(int(histogram.sum()) == len(values), "Histogram silently excludes samples")
        result["histogram_counts"] = histogram.tolist()
    return result


def summarize(records, references, bins):
    n = len(records)
    require(n > 0, "No eligible routes to summarize")
    pair_total = sum(r["adjacent_pairs"] for r in records)
    invalid = sum(r["invalid_transitions"] for r in records)
    total_roads = sum(r["road_count"] for r in records)
    result = {"samples": n, "reference_trajectories": len(references),
              "road_count": describe([r["road_count"] for r in records], bins["road_count"]),
              "distance_m": describe([r["distance_m"] for r in records if r["distance_m"] is not None], bins["distance_m"]),
              "missing_distance_samples": sum(r["distance_m"] is None for r in records),
              "immediate_stop_rate": sum(r["immediate_stop"] for r in records) / n,
              "length_cap_rate": sum(r["termination"] == "length_cap" for r in records) / n,
              "termination_counts": dict(Counter(r["termination"] for r in records)),
              "valid_road_id_token_rate": sum(r["valid_road_tokens"] for r in records) / total_roads,
              "all_ids_valid_route_rate": sum(r["valid_road_ids"] for r in records) / n,
              "directed_adjacent_pair_connectivity_rate": 1 - invalid / pair_total if pair_total else None,
              "adjacent_pairs": pair_total, "invalid_transitions": invalid,
              "fraction_samples_with_invalid_transition": sum(r["has_invalid_transition"] for r in records) / n,
              "single_road_samples": sum(r["single_road"] for r in records),
              "multi_road_samples": sum(not r["single_road"] for r in records),
              "repeated_road_route_rate": sum(r["has_repeated_road"] for r in records) / n,
              "repeated_occurrence_fraction": sum(r["repeated_occurrences"] for r in records) / total_roads,
              "per_route_unique_road_fraction": describe([r["unique_road_fraction"] for r in records]),
              "exact_unique_routes": len({tuple(r["roads"]) for r in records}),
              "exact_route_uniqueness": len({tuple(r["roads"]) for r in records}) / n,
              "distinct_origins": len({r["roads"][0] for r in records}),
              "distinct_prompts": len({tuple(r["prompt_roads"]) for r in records})}
    for metric in ("road_count", "distance_m"):
        ref = describe([r[metric] for r in references], bins[metric])
        result["reference_" + metric] = ref
        result[metric + "_js_distance"] = float(jensenshannon(
            result[metric]["histogram_counts"], ref["histogram_counts"], base=2))
    sensitivity = {}
    for width in (100, 250, 500):
        end = max(bins["distance_m"][-1], width)
        alternate = np.arange(0, math.ceil(end / width) * width + width, width)
        generated = np.histogram([r["distance_m"] for r in records if r["distance_m"] is not None], alternate)[0]
        reference = np.histogram([r["distance_m"] for r in references], alternate)[0]
        sensitivity[str(width)] = float(jensenshannon(generated, reference, base=2))
    result["distance_js_bin_width_sensitivity_m"] = sensitivity
    return result


def verify_pairing(records, selected, checkpoints=CHECKPOINTS, seeds=SEEDS):
    expected = {(c, r["trajectory_id"], p, s) for c in checkpoints for r in selected
                for p in r["eligible_prompt_lengths"] for s in seeds}
    actual = [(r["checkpoint"], r["trajectory_id"], r["prompt_length"], r["seed"]) for r in records]
    require(len(actual) == len(set(actual)) and set(actual) == expected, "Missing, extra, or duplicate generated samples")
    lookup = {r["trajectory_id"]: r for r in selected}
    for row in records:
        source = lookup[row["trajectory_id"]]
        require(row["parent_id"] == source["parent_id"], "Parent identity mismatch")
        require(row["prompt_roads"] == source["reference_roads"][:row["prompt_length"]], "Prompt mismatch")
        require(row["derived_seed"] == stable_seed(row["seed"], row["trajectory_id"], row["prompt_length"]), "Seed mismatch")
        require(row["roads"][:row["prompt_length"]] == row["prompt_roads"], "Generated prompt was changed")
        require(row["termination"] in ("eos", "length_cap"), "Unknown termination")
        require(row["termination"] != "length_cap" or row["road_count"] == 80, "Incorrect length cap accounting")
        require(row["road_count"] == len(row["roads"]) and row["road_count"] >= row["prompt_length"], "Road count mismatch")
        require(row["immediate_stop"] == (row["road_count"] == row["prompt_length"]), "Immediate stop accounting mismatch")
    return {"passed": True, "expected_samples": len(expected), "actual_samples": len(records),
            "paired_cases": len(expected) // len(checkpoints), "silently_dropped_samples": 0,
            "matched_prompts_seeds_and_parents": True, "termination_accounting": "passed"}


