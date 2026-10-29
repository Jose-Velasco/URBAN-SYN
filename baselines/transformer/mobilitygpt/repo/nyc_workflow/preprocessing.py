#!/usr/bin/env python3
"""Conservative NYC cleaning/export; the standard-library scientific implementation.

Internal CLI invoked by the installed workflow; all data/output paths are explicit.
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LOG = logging.getLogger("prepare_nyc")
from .runtime import require_output, module_command
csv.field_size_limit(sys.maxsize)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def road_id(value, context):
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            f"{context}: malformed road ID {value!r}; expected a nonnegative integer")
    return int(value)


def read_csv(path, required):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None and set(required) <= set(reader.fieldnames),
                f"{path}: missing required columns {sorted(required)}")
        require(len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"{path}: duplicate column names")
        for line, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()),
                    f"{path}:{line}: malformed CSV row")
            yield row


def header(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


def write_csv(path, columns, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


@dataclass
class Node:
    road: int
    start: object
    stop: object
    time: str

    def as_dict(self):
        return {"road_id": self.road,
                "observation_range": None if self.start is None else [self.start, self.stop],
                "timestamp": self.time}


@dataclass
class Trip:
    parent: str
    observations: int
    nodes: list
    sections: list
    gaps: list
    repairs: list
    repeated: int


def load_network(source):
    geo = source / "nyc.geo"
    roads = set()
    for row in read_csv(geo, {"geo_id"}):
        rid = road_id(row["geo_id"], str(geo))
        require(rid not in roads, f"{geo}: duplicate geo_id {rid}")
        roads.add(rid)
    require(roads, f"{geo}: empty geography")
    outgoing, incoming = defaultdict(set), defaultdict(set)
    relations = 0
    for row in read_csv(source / "nyc.rel", {"origin_id", "destination_id"}):
        a = road_id(row["origin_id"], "relation origin")
        b = road_id(row["destination_id"], "relation destination")
        require(a in roads and b in roads, f"nyc.rel: unknown endpoint {a} -> {b}")
        outgoing[a].add(b)
        incoming[b].add(a)
        relations += 1
    return roads, outgoing, incoming, relations


def clean_trip(parent, ids, times, outgoing, incoming):
    """Only bridge original collapsed neighbors; never recursively repair gaps."""
    collapsed = []
    for index, (rid, timestamp) in enumerate(zip(ids, times)):
        if collapsed and collapsed[-1].road == rid:
            collapsed[-1].stop = index + 1
        else:
            collapsed.append(Node(rid, index, index + 1, timestamp))
    require(collapsed, f"trip {parent}: empty trajectory")
    nodes = [collapsed[0]]
    sections, gaps, repairs = [], [], []
    section_start = 0
    for left, right in zip(collapsed, collapsed[1:]):
        a, c = left.road, right.road
        if c not in outgoing.get(a, ()):
            connectors = outgoing.get(a, set()) & incoming.get(c, set())
            boundary = {"from_road": a, "to_road": c,
                        "left_observation_index": left.stop - 1,
                        "right_observation_index": right.start}
            if len(connectors) == 1:
                connector = next(iter(connectors))
                repairs.append(dict(boundary, inserted_road=connector, node_index=len(nodes)))
                nodes.append(Node(connector, None, None, ""))
            else:
                sections.append((section_start, len(nodes)))
                section_start = len(nodes)
                gaps.append(dict(boundary, node_index=len(nodes),
                                 reason="ambiguous" if connectors else "unresolved",
                                 connector_candidates=sorted(connectors)))
        nodes.append(right)
    sections.append((section_start, len(nodes)))
    return Trip(parent, len(ids), nodes, sections, gaps, repairs, len(ids) - len(collapsed))


def load_trips(path, roads, outgoing, incoming, forbidden_parents=()):
    parents, trips = set(), []
    for row in read_csv(path, {"traj_id", "rid_list", "time_list"}):
        parent = row["traj_id"]
        context = f"{path}, trip {parent!r}"
        require(parent and parent == parent.strip(), f"{context}: malformed parent trip ID")
        require(parent not in parents, f"{context}: duplicate parent trip ID")
        require(parent not in forbidden_parents, f"{context}: parent appears in train and test")
        parents.add(parent)
        ids = [road_id(value, context) for value in row["rid_list"].split(",")]
        unknown = set(ids) - roads
        require(not unknown, f"{context}: unknown road IDs {sorted(unknown)[:10]}")
        times = row["time_list"].split(",")
        require(len(ids) == len(times) and all(times),
                f"{context}: road/time length mismatch or missing source timestamp")
        trips.append(clean_trip(parent, ids, times, outgoing, incoming))
    require(trips, f"{path}: no trajectories")
    return trips, parents


def rank_roads(trips):
    frequency = Counter()
    for trip in trips:
        # Each source trip casts at most one vote for a road, even across splits.
        frequency.update({node.road for start, stop in trip.sections if stop - start >= 2
                          for node in trip.nodes[start:stop]})
    return sorted(frequency, key=lambda rid: (-frequency[rid], rid)), frequency


def within(trip, start, stop, network):
    return stop - start >= 2 and all(n.road in network for n in trip.nodes[start:stop])


def choose_network(trips, limit):
    ranking, frequency = rank_roads(trips)
    candidates = set(ranking[:limit])
    used = {n.road for t in trips for start, stop in t.sections
            if within(t, start, stop, candidates) for n in t.nodes[start:stop]}
    require(used, "No complete training sections survive the candidate limit; increase it")
    return {rid: index for index, rid in enumerate(sorted(used))}, ranking, frequency, candidates


def observation_count(nodes):
    return sum(n.stop - n.start for n in nodes if n.start is not None)


TRAJECTORY_COLUMNS = ["traj_id", "parent_traj_id", "section_index", "rid_list", "time_list",
                      "original_rid_list", "observation_ranges"]


def export_trips(directory, name, split, trips, mapping, text_handle=None):
    counts = Counter({key: 0 for key in (
        "original_trips", "original_observations", "collapsed_observed_road_runs",
        "repeated_observations_collapsed", "unique_repairs", "ambiguous_gaps",
        "unresolved_gaps", "split_boundaries", "trips_split", "sections_before_min_length",
        "singleton_sections_discarded", "singleton_observations_discarded",
        "cleaned_sections", "cleaned_observations", "cleaned_parent_trips",
        "sections_lost_to_network", "observations_lost_to_network", "parents_lost_to_network",
        "retained_sections", "retained_original_trips", "retained_observations",
        "retained_road_tokens", "retained_inserted_connectors", "retained_complete_trips",
        "retained_sections_from_split_trips", "retained_split_parent_trips",
        "original_trips_lost_total", "original_observations_lost_total")})
    max_length = 0
    path = directory / f"{name}_Taxi_trajectory_{split}.csv"
    with path.open("x", newline="", encoding="utf-8") as csv_handle, \
            (directory / f"{split}_provenance.jsonl").open("x", encoding="utf-8") as provenance:
        writer = csv.DictWriter(csv_handle, fieldnames=TRAJECTORY_COLUMNS)
        writer.writeheader()
        for trip in trips:
            counts["original_trips"] += 1
            counts["original_observations"] += trip.observations
            counts["repeated_observations_collapsed"] += trip.repeated
            counts["collapsed_observed_road_runs"] += trip.observations - trip.repeated
            counts["unique_repairs"] += len(trip.repairs)
            counts["split_boundaries"] += len(trip.gaps)
            counts["trips_split"] += bool(trip.gaps)
            counts["ambiguous_gaps"] += sum(g["reason"] == "ambiguous" for g in trip.gaps)
            counts["unresolved_gaps"] += sum(g["reason"] == "unresolved" for g in trip.gaps)
            counts["sections_before_min_length"] += len(trip.sections)
            sections, retained, cleaned = [], 0, 0
            for section_index, (start, stop) in enumerate(trip.sections):
                nodes = trip.nodes[start:stop]
                observations = observation_count(nodes)
                entry = {"section_index": section_index, "node_range": [start, stop],
                         "observations": observations, "export_traj_id": None}
                if len(nodes) < 2:
                    entry["status"] = "discarded_singleton"
                    counts["singleton_sections_discarded"] += 1
                    counts["singleton_observations_discarded"] += observations
                else:
                    cleaned += 1
                    counts["cleaned_sections"] += 1
                    counts["cleaned_observations"] += observations
                    if not within(trip, start, stop, mapping):
                        entry["status"] = "discarded_network"
                        counts["sections_lost_to_network"] += 1
                        counts["observations_lost_to_network"] += observations
                    else:
                        retained += 1
                        entry.update(status="retained", export_traj_id=f"{trip.parent}:{section_index}")
                        new_ids = ",".join(str(mapping[n.road]) for n in nodes)
                        writer.writerow({"traj_id": entry["export_traj_id"],
                                         "parent_traj_id": trip.parent, "section_index": section_index,
                                         "rid_list": new_ids, "time_list": ",".join(n.time for n in nodes),
                                         "original_rid_list": ",".join(str(n.road) for n in nodes),
                                         "observation_ranges": json.dumps(
                                             [n.as_dict()["observation_range"] for n in nodes])})
                        if text_handle is not None:
                            text_handle.write(new_ids + "\n\n")
                        counts["retained_sections"] += 1
                        counts["retained_observations"] += observations
                        counts["retained_road_tokens"] += len(nodes)
                        counts["retained_inserted_connectors"] += sum(n.start is None for n in nodes)
                        counts["retained_sections_from_split_trips"] += bool(trip.gaps)
                        max_length = max(max_length, len(nodes))
                sections.append(entry)
            counts["cleaned_parent_trips"] += bool(cleaned)
            counts["parents_lost_to_network"] += bool(cleaned and not retained)
            counts["retained_original_trips"] += bool(retained)
            counts["retained_split_parent_trips"] += bool(retained and trip.gaps)
            counts["retained_complete_trips"] += bool(retained and not trip.gaps)
            record = {"split": split, "parent_traj_id": trip.parent,
                      "original_observations": trip.observations,
                      "repeated_observations_collapsed": trip.repeated,
                      "nodes": [n.as_dict() for n in trip.nodes],
                      "repairs": trip.repairs, "split_boundaries": trip.gaps, "sections": sections}
            provenance.write(json.dumps(record, separators=(",", ":")) + "\n")
    counts["original_trips_lost_total"] = counts["original_trips"] - counts["retained_original_trips"]
    counts["original_observations_lost_total"] = counts["original_observations"] - counts["retained_observations"]
    require(counts["original_observations_lost_total"] ==
            counts["singleton_observations_discarded"] + counts["observations_lost_to_network"],
            f"{split}: observation accounting mismatch")
    result = dict(counts, max_retained_section_roads=max_length)
    for label, numerator, denominator in (
            ("parent_trip_retention_fraction", "retained_original_trips", "original_trips"),
            ("cleaned_section_retention_fraction", "retained_sections", "cleaned_sections"),
            ("observation_retention_fraction", "retained_observations", "original_observations")):
        result[label] = counts[numerator] / counts[denominator] if counts[denominator] else 0
    LOG.info("%s: %d sequences, %d parents, %d complete trips; %d singleton sections discarded",
             split, counts["retained_sections"], counts["retained_original_trips"],
             counts["retained_complete_trips"], counts["singleton_sections_discarded"])
    return result


def verify_export(directory, text_path, name):
    """Read back the artifacts, including provenance for discarded sections."""
    rows = list(read_csv(directory / "id_mapping.csv", {"original_id", "new_id"}))
    originals = [road_id(r["original_id"], "mapping") for r in rows]
    new_ids = [road_id(r["new_id"], "mapping") for r in rows]
    require(originals == sorted(set(originals)) and new_ids == list(range(len(rows))),
            "ID mapping must be unique and consecutive in ascending original-ID order")
    mapping = dict(zip(originals, new_ids))
    geo_ids = [road_id(r["geo_id"], "export geography")
               for r in read_csv(directory / "roadmap.geo", {"geo_id"})]
    require(geo_ids == new_ids and geo_ids, "Geography IDs do not match row positions")
    edges = set()
    for row in read_csv(directory / "roadmap.rel", {"origin_id", "destination_id"}):
        a, b = road_id(row["origin_id"], "export relation"), road_id(row["destination_id"], "export relation")
        require(0 <= a < len(rows) and 0 <= b < len(rows), "Relation endpoint outside vocabulary")
        edges.add((a, b))
    require(edges and max(max(pair) for pair in edges) == len(rows) - 1,
            "Highest geography ID must occur in relations for adjacency sizing")
    vocabulary_ids = set(new_ids)
    all_parents, train_sequences, used_train, pair_count = set(), [], set(), 0
    for split in ("train", "test"):
        exported = {}
        for row in read_csv(directory / f"{name}_Taxi_trajectory_{split}.csv", set(TRAJECTORY_COLUMNS)):
            require(row["traj_id"] not in exported, "Duplicate exported section ID")
            exported[row["traj_id"]] = row
            ids = [road_id(x, "export trajectory") for x in row["rid_list"].split(",")]
            require(len(ids) >= 2 and set(ids) <= vocabulary_ids, "Invalid trajectory vocabulary")
            require(all((a, b) in edges for a, b in zip(ids, ids[1:])),
                    f"{split} {row['traj_id']}: adjacent pair missing from relations")
            pair_count += len(ids) - 1
            if split == "train":
                train_sequences.append(row["rid_list"])
                used_train.update(ids)
        found, split_parents = set(), set()
        with (directory / f"{split}_provenance.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                trip = json.loads(line)
                parent = trip["parent_traj_id"]
                require(trip["split"] == split and parent not in all_parents and parent not in split_parents,
                        "Original parents overlap across train/test or are duplicated")
                split_parents.add(parent)
                nodes = trip["nodes"]
                cursor = 0
                for node in nodes:
                    span = node["observation_range"]
                    if span is None:
                        require(node["timestamp"] == "", "Invented timestamp for connector")
                    else:
                        require(span[0] == cursor and span[1] > cursor and node["timestamp"],
                                "Invalid original observation provenance")
                        cursor = span[1]
                require(cursor == trip["original_observations"], "Observations missing from provenance")
                cursor = 0
                for section in trip["sections"]:
                    start, stop = section["node_range"]
                    require(start == cursor and start < stop <= len(nodes), "Invalid section provenance")
                    cursor = stop
                    section_nodes = nodes[start:stop]
                    require(section["observations"] == sum(n["observation_range"][1] - n["observation_range"][0]
                            for n in section_nodes if n["observation_range"] is not None),
                            "Incorrect section observation count")
                    rid = section["export_traj_id"]
                    if section["status"] != "retained":
                        require(rid is None, "Discarded section has an export ID")
                        continue
                    require(rid in exported and rid not in found, "Missing or duplicate section provenance")
                    row = exported[rid]
                    found.add(rid)
                    require(row["parent_traj_id"] == parent and int(row["section_index"]) == section["section_index"],
                            "Incorrect section parent")
                    require(row["original_rid_list"] == ",".join(str(n["road_id"]) for n in section_nodes),
                            "Original roads differ from provenance")
                    require(row["rid_list"] == ",".join(str(mapping[n["road_id"]]) for n in section_nodes),
                            "Exported roads differ from provenance")
                    require(row["time_list"] == ",".join(n["timestamp"] for n in section_nodes),
                            "Timestamps differ from provenance")
                    require(json.loads(row["observation_ranges"]) == [n["observation_range"] for n in section_nodes],
                            "Observation ranges differ from provenance")
                require(cursor == len(nodes), "Nodes missing from section provenance")
        require(found == set(exported), "Exported section missing provenance")
        all_parents.update(split_parents)
    require(used_train == set(new_ids), "Geography contains roads unused by training sections")
    # This is precisely the blank-line/comma parsing used by CharDataset.
    raw = text_path.read_text(encoding="utf-8")
    sequences = raw.strip().split("\n\n")
    require(sequences == train_sequences, "Text sequences differ from exported training CSV")
    vocab = set(map(str, new_ids))
    require(all(set(s.strip().split(",")) <= vocab for s in sequences), "Text token outside vocabulary")
    return {"passed": True, "roads": len(rows), "distinct_relations": len(edges),
            "adjacent_pairs_checked": pair_count, "training_sequences": len(sequences),
            "original_parents_checked": len(all_parents), "adjacency_dimension_with_eos": len(rows) + 1,
            "char_dataset_parsing": "passed; actual class also checked by --measure-adjacency"}


def adjacency_worker(root, name, device):
    """Isolate builder peak RSS from preprocessing and insist on requested CUDA."""
    import resource
    import time
    from types import SimpleNamespace

    import torch
    from datasets.char_dataset import CharDataset
    from datasets.data_utils import load_geo_data, od_pair_to_adjacency_matrix

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    is_cuda = device.startswith("cuda")
    require(not is_cuda or torch.cuda.is_available(),
            "CUDA was requested but is unavailable. Check GPU access and the CUDA PyTorch installation")
    if is_cuda:
        requested = torch.device(device)
        device = f"cuda:{requested.index if requested.index is not None else torch.cuda.current_device()}"
    geo, vocab, rel, pairs = load_geo_data(name, root)
    dataset = CharDataset(SimpleNamespace(block_size=81), (root/f"Trajs_{name}.txt").read_text(), vocab)
    require(all(token in dataset.stoi for token in dataset.data), "CharDataset token outside vocabulary")
    dataset[0]
    info = {"device": device, "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "char_dataset_sequences": dataset.num_trajs, "training_started": False,
            "baseline_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if is_cuda:
        torch.cuda.set_device(device)
        # Exercise CUDA as well as allocating adjacency directly in VRAM.
        probe = torch.ones((128, 128), device=device)
        require((probe @ probe)[0, 0].item() == 128, "CUDA matrix multiplication failed")
        del probe
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()
        info.update(gpu_name=torch.cuda.get_device_name(), gpu_free_before_bytes=free, gpu_total_bytes=total,
                    cuda_matrix_multiply="passed")
    started = time.monotonic()
    matrix = od_pair_to_adjacency_matrix(pairs, device)
    if is_cuda:
        torch.cuda.synchronize()
    require(tuple(matrix.shape) == (len(geo) + 1, len(geo) + 1), "Actual adjacency dimensions incorrect")
    require(matrix.device.type == ("cuda" if is_cuda else "cpu"), "Adjacency is on the wrong device")
    require(matrix.dtype == torch.bool, "Adjacency must use boolean membership storage")
    origins = torch.tensor([p[0] for p in pairs], device=device)
    destinations = torch.tensor([p[1] for p in pairs], device=device)
    require(bool(torch.all(matrix[origins, destinations] == 1).item()), "Actual adjacency lost relations")
    require(bool(torch.all(matrix[-1, :] == 1).item()) and bool(torch.all(matrix[:, -1] == 1).item()),
            "Actual adjacency EOS row/column incorrect")
    info.update(shape=list(matrix.shape), dtype=str(matrix.dtype),
                final_tensor_bytes=matrix.numel() * matrix.element_size(),
                direct_boolean_allocation_bytes=(len(geo) + 1) ** 2,
                allocation_strategy="final boolean shape allocated directly on requested device",
                process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                elapsed_seconds=time.monotonic() - started)
    if is_cuda:
        info.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print(json.dumps(info))


def measure_adjacency(root, name, device):
    command = [*module_command("nyc_workflow.preprocessing"), "--_adjacency-worker",
               "--output-root", str(root), "--dataset-name", name, "--device", device]
    result = subprocess.run(command, capture_output=True, text=True)
    require(result.returncode == 0, "Adjacency measurement failed:\n" + result.stdout + result.stderr)
    return json.loads(result.stdout.strip().splitlines()[-1])


def summary_text(meta):
    lines = [f"{meta['dataset_name']} preprocessing report", "",
             f"Candidate roads: {meta['network']['candidate_roads']:,}",
             f"Final roads: {meta['network']['final_roads']:,}",
             "Selection: distinct training parents per road; ties by ascending original ID.",
             "Only whole sections are retained. Test trips never select roads.", ""]
    for split, stats in meta["splits"].items():
        lines.extend([split.upper(), *[f"  {key}: {value}" for key, value in stats.items()], ""])
    lines.extend(["Observation retention counts all raw observations represented by collapsed run ranges.",
                  "Repeated observations are represented, not counted as discarded observations.",
                  "Complete trips have no unresolved/ambiguous split and retain their whole cleaned path.",
                  "Singleton sections are discarded before road ranking.",
                  "Provenance includes every parent, repair, split, singleton and network rejection.",
                  "Ranges are zero-based [start, stop); source timestamps remain in the immutable input.",
                  "CSV timestamps use the first observation in each run; inserted roads have empty timestamps.",
                  "No sequences are truncated at the model's 81-road block size.", "",
                  "Adjacency memory (bytes):", json.dumps(meta["adjacency"], indent=2, sort_keys=True), "",
                  "Validation:", json.dumps(meta["validation"], indent=2, sort_keys=True), "",
                  "Training was not started. Adjacency remains enabled in the unchanged model configuration.",
                  "Gravity pretraining requires validated dataset-specific artifacts and a trajectory adapter.",
                  "Measure model/training memory separately before a full run."])
    return "\n".join(lines) + "\n"


def prepare(source, root, name, limit, measure=False, device="cuda"):
    source = Path(source).resolve()
    root = require_output(root)
    require_output(root/f"{name}-Taxi", [source])
    require_output(root/f"Trajs_{name}.txt", [source])
    require(limit > 0, "Candidate road limit must be positive")
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is not None,
            "Dataset name must contain only letters, numbers, underscores and hyphens")
    directory, text_path = root / f"{name}-Taxi", root / f"Trajs_{name}.txt"
    require(not directory.exists() and not text_path.exists(),
            f"Refusing to overwrite outputs: {directory} or {text_path}")
    inputs = {key: source / file for key, file in (
        ("geography", "nyc.geo"), ("relations", "nyc.rel"),
        ("train", "nyc_mm_train.csv"), ("test", "nyc_mm_test.csv"))}
    fingerprints = {key: fingerprint(path) for key, path in inputs.items()}
    roads, outgoing, incoming, relation_count = load_network(source)
    LOG.info("Loaded %d roads and %d directed relations", len(roads), relation_count)
    train, train_parents = load_trips(inputs["train"], roads, outgoing, incoming)
    mapping, ranking, frequency, candidates = choose_network(train, limit)
    LOG.info("Selected %d candidates; %d roads used by retained training sections", len(candidates), len(mapping))
    test, _ = load_trips(inputs["test"], roads, outgoing, incoming, train_parents)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".prepare-{name}-", dir=root) as temporary:
        staging_root = Path(temporary)
        stage = staging_root / directory.name
        stage.mkdir()
        staged_text = staging_root / text_path.name
        def geo_rows():
            selected = {road_id(r["geo_id"], "geography"): r
                        for r in read_csv(inputs["geography"], {"geo_id"})
                        if road_id(r["geo_id"], "geography") in mapping}
            for rid in mapping:
                yield dict(selected[rid], geo_id=mapping[rid])
        write_csv(stage / "roadmap.geo", header(inputs["geography"]), geo_rows())
        def relation_rows():
            for row in read_csv(inputs["relations"], {"origin_id", "destination_id"}):
                a, b = int(row["origin_id"]), int(row["destination_id"])
                if a in mapping and b in mapping:
                    yield dict(row, origin_id=mapping[a], destination_id=mapping[b])
        write_csv(stage / "roadmap.rel", header(inputs["relations"]), relation_rows())
        write_csv(stage / "id_mapping.csv", ["original_id", "new_id", "training_parent_frequency"],
                  ({"original_id": rid, "new_id": new, "training_parent_frequency": frequency[rid]}
                   for rid, new in mapping.items()))
        write_csv(stage / "road_ranking.csv", ["rank", "original_id", "training_parent_frequency", "candidate", "retained"],
                  ({"rank": rank, "original_id": rid, "training_parent_frequency": frequency[rid],
                    "candidate": int(rid in candidates), "retained": int(rid in mapping)}
                   for rank, rid in enumerate(ranking, 1)))
        with staged_text.open("x", encoding="utf-8") as handle:
            train_counts = export_trips(stage, name, "train", train, mapping, handle)
        test_counts = export_trips(stage, name, "test", test, mapping)
        validation = verify_export(stage, staged_text, name)
        require(fingerprints == {key: fingerprint(path) for key, path in inputs.items()},
                "Source files changed during processing")
        adjacency = {"measured": False, "requested_device": device,
                     "dtype": "torch.bool",
                     "expected_final_boolean_bytes": (len(mapping) + 1) ** 2,
                     "allocation_strategy": "direct final shape; includes EOS row and column",
                     "note": "Estimates exclude transient allocations and training; use --measure-adjacency"}
        if measure:
            LOG.info("Measuring the existing adjacency builder on %s in a fresh process", device)
            adjacency = dict(measure_adjacency(staging_root, name, device), measured=True)
        meta = {"schema_version": 1, "dataset_name": name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": fingerprint(Path(__file__)), "source_files": fingerprints,
                "parameters": {"candidate_road_limit": limit, "repair": "unique_existing_two_hop_only",
                               "minimum_section_roads": 2, "ranking": "distinct_cleaned_training_parents_then_original_id",
                               "observation_ranges": "zero_based_half_open", "sequence_truncation": None},
                "network": {"source_roads": len(roads), "source_relation_rows": relation_count,
                            "ranked_training_roads": len(ranking), "candidate_roads": len(candidates),
                            "unused_candidates_removed": len(candidates) - len(mapping), "final_roads": len(mapping)},
                "splits": {"train": train_counts, "test": test_counts},
                "adjacency": adjacency, "validation": validation,
                "training": {"started": False, "adjacency_enabled_by_default": True,
                             "gravity_artifacts_generated": False,
                             "runtime_sampling_behavior_requires_review": True}}
        write_json(stage / "processing_metadata.json", meta)
        (stage / "summary_report.txt").write_text(summary_text(meta), encoding="utf-8")
        # Publish only validated artifacts; exclusive text creation also guards reruns.
        require(not directory.exists(), f"Refusing to overwrite {directory}")
        os.link(staged_text, text_path)
        try:
            stage.rename(directory)
        except BaseException:
            text_path.unlink()  # Only the link created by this invocation.
            raise
    LOG.info("Validated dataset: %s; training text: %s", directory, text_path)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, help="Required when preparing a new export")
    parser.add_argument("--dataset-name", "--dataset", default="NYC20K")
    parser.add_argument("--candidate-road-limit", "--candidate-limit", type=int, default=20000)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true", help="Verify existing exports without regenerating them")
    parser.add_argument("--measure-adjacency", action="store_true", help="Measure the actual builder and CharDataset in a subprocess")
    parser.add_argument("--device", default="cuda", help="Adjacency measurement device; default cuda (no CPU fallback)")
    parser.add_argument("--_adjacency-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.dataset_name) is not None, "Invalid dataset name")
        root = args.output_root.resolve()
        if args._adjacency_worker:
            adjacency_worker(root, args.dataset_name, args.device)
        elif args.verify_only:
            directory = root / f"{args.dataset_name}-Taxi"
            validation = verify_export(directory, root / f"Trajs_{args.dataset_name}.txt", args.dataset_name)
            print(json.dumps(validation, indent=2))
            if args.measure_adjacency:
                result = dict(measure_adjacency(root, args.dataset_name, args.device), measured=True)
                # A separate measurement file preserves the original processing record.
                target = directory / f"adjacency_measurement_{args.device.replace(':', '_')}.json"
                write_json(target, result)
                print(json.dumps(result, indent=2))
        else:
            require(args.source_dir is not None, "--source-dir is required for preparation")
            prepare(args.source_dir.resolve(), root, args.dataset_name, args.candidate_road_limit,
                    args.measure_adjacency, args.device)
    except (ValueError, OSError, csv.Error) as exc:
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
