"""Small deterministic fixtures; run python -m unittest discover -s preprocessing -v."""

import csv
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from nyc_workflow.preprocessing import (choose_network, clean_trip, fingerprint, load_trips, prepare,
                         rank_roads, road_id, verify_export, write_csv)


def graph(edges):
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for a, b in edges:
        outgoing[a].add(b)
        incoming[b].add(a)
    return outgoing, incoming


def trip(parent, ids, edges):
    return clean_trip(parent, ids, [f"t{i}" for i in range(len(ids))], *graph(edges))


class CleaningTests(unittest.TestCase):
    def test_unique_repair_and_original_indices(self):
        cleaned = trip("a", [1, 1, 3, 3], [(1, 2), (1, 2), (2, 3)])
        self.assertEqual([n.road for n in cleaned.nodes], [1, 2, 3])
        self.assertEqual([(n.start, n.stop) for n in cleaned.nodes], [(0, 2), (None, None), (2, 4)])
        self.assertEqual([n.time for n in cleaned.nodes], ["t0", "", "t2"])
        self.assertEqual(cleaned.sections, [(0, 3)])
        self.assertEqual(cleaned.repairs[0]["left_observation_index"], 1)
        self.assertEqual(cleaned.repairs[0]["right_observation_index"], 2)

    def test_ambiguous_and_unresolved_gaps_split_without_repair(self):
        edges = [(1, 2), (2, 4), (1, 3), (3, 4), (4, 5)]
        cleaned = trip("a", [1, 4, 5, 8], edges)
        self.assertEqual(cleaned.sections, [(0, 1), (1, 3), (3, 4)])
        self.assertEqual([g["reason"] for g in cleaned.gaps], ["ambiguous", "unresolved"])
        self.assertEqual(cleaned.gaps[0]["connector_candidates"], [2, 3])
        self.assertEqual(cleaned.repairs, [])

    def test_consecutive_duplicates_only(self):
        cleaned = trip("a", [1, 1, 2, 2, 1], [(1, 2), (2, 1)])
        self.assertEqual([n.road for n in cleaned.nodes], [1, 2, 1])
        self.assertEqual(cleaned.repeated, 2)

    def test_ranking_counts_parents_once_across_sections_and_revisits(self):
        edges = [(1, 2), (2, 1), (3, 4)]
        a = trip("a", [1, 2, 1, 9, 3, 4], edges)
        b = trip("b", [3, 4], edges)
        ranking, frequency = rank_roads([a, b])
        self.assertEqual(ranking, [3, 4, 1, 2])
        self.assertEqual(frequency, {1: 1, 2: 1, 3: 2, 4: 2})
        self.assertEqual(rank_roads([b, a]), (ranking, frequency))
        self.assertNotIn(9, frequency)  # singleton was discarded before ranking

    def test_inserted_connectors_receive_training_votes(self):
        a = trip("a", [1, 3], [(1, 2), (2, 3)])
        self.assertEqual(rank_roads([a])[1], {1: 1, 2: 1, 3: 1})

    def test_selection_retains_whole_sections_and_prunes_unused_candidates(self):
        edges = [(1, 2), (2, 3), (4, 5)]
        trips = [trip("a", [1, 2, 3], edges), trip("b", [1, 2, 3], edges),
                 trip("c", [4, 5], edges)]
        mapping, _, _, candidates = choose_network(trips, 4)
        self.assertEqual(candidates, {1, 2, 3, 4})
        self.assertEqual(mapping, {1: 0, 2: 1, 3: 2})
        with self.assertRaisesRegex(ValueError, "No complete training sections"):
            choose_network(trips, 1)

    def test_malformed_road_ids(self):
        for value in ("1.0", "-1", "", " 1", "NaN", "1e2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "malformed road ID"):
                road_id(value, "fixture")


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        write_csv(self.source / "nyc.geo", ["geo_id", "coordinates", "length", "custom"],
                  ({"geo_id": rid, "coordinates": "[[0,0],[1,1]]", "length": "1.0", "custom": f"attr-{rid}"}
                   for rid in range(1, 9)))
        write_csv(self.source / "nyc.rel", ["rel_id", "type", "origin_id", "destination_id"],
                  ({"rel_id": i, "type": "geo", "origin_id": a, "destination_id": b}
                   for i, (a, b) in enumerate([(1, 2), (2, 3), (3, 1), (4, 5), (6, 7)])))
        # a: repair 1->3; b: singleton 8 plus valid 4->5; c: long cycle.
        self.write_trips("train", [("a", [1, 1, 3]), ("b", [8, 4, 5]), ("c", [1, 2, 3] * 30)])
        self.write_trips("test", [("test-a", [1, 3]), ("test-b", [6, 7])])

    def write_trips(self, split, values):
        write_csv(self.source / f"nyc_mm_{split}.csv", ["traj_id", "rid_list", "time_list"],
                  ({"traj_id": parent, "rid_list": ",".join(map(str, ids)),
                    "time_list": ",".join(f"time-{i}" for i in range(len(ids)))}
                   for parent, ids in values))

    def test_end_to_end_provenance_retention_and_no_overwrite(self):
        before = {p.name: fingerprint(p) for p in self.source.iterdir()}
        meta = prepare(self.source, self.root, "Fixture", 4)
        directory = self.root / "Fixture-Taxi"
        self.assertEqual(meta["network"]["final_roads"], 3)
        train, test = meta["splits"]["train"], meta["splits"]["test"]
        self.assertEqual(train["retained_sections"], 2)
        self.assertEqual(train["retained_complete_trips"], 2)
        self.assertEqual(train["singleton_sections_discarded"], 1)
        self.assertEqual(train["sections_lost_to_network"], 1)
        self.assertEqual(train["singleton_observations_discarded"], 1)
        self.assertEqual(train["observations_lost_to_network"], 2)
        self.assertEqual(test["retained_sections"], 1)
        self.assertEqual(test["sections_lost_to_network"], 1)
        self.assertEqual(train["max_retained_section_roads"], 90)
        self.assertEqual(before, {p.name: fingerprint(p) for p in self.source.iterdir()})
        self.assertTrue(verify_export(directory, self.root / "Trajs_Fixture.txt", "Fixture")["passed"])
        with (directory / "roadmap.geo").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["custom"] for r in rows], ["attr-1", "attr-2", "attr-3"])
        provenance = [json.loads(line) for line in (directory / "train_provenance.jsonl").read_text().splitlines()]
        self.assertEqual(provenance[0]["nodes"][1]["observation_range"], None)
        self.assertEqual(provenance[1]["sections"][0]["status"], "discarded_singleton")
        self.assertEqual(provenance[1]["sections"][1]["status"], "discarded_network")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            prepare(self.source, self.root, "Fixture", 4)

    def test_unknown_ids_and_timestamp_length_fail(self):
        outgoing, incoming = graph([(1, 2)])
        for ids, times, message in [("1,999", "a,b", "unknown road"),
                                    ("1,2", "a", "length mismatch"),
                                    ("1.0,2", "a,b", "malformed road ID")]:
            path = self.root / f"bad-{message}.csv"
            write_csv(path, ["traj_id", "rid_list", "time_list"],
                      [{"traj_id": "bad", "rid_list": ids, "time_list": times}])
            with self.assertRaisesRegex(ValueError, message):
                load_trips(path, {1, 2}, outgoing, incoming)

    def test_parent_leakage_fails_before_output(self):
        (self.source / "nyc_mm_test.csv").unlink()
        self.write_trips("test", [("a", [1, 2])])
        with self.assertRaisesRegex(ValueError, "parent appears in train and test"):
            prepare(self.source, self.root, "Leak", 8)
        self.assertFalse((self.root / "Leak-Taxi").exists())

    def test_highest_id_and_pair_validation_detect_corruption(self):
        prepare(self.source, self.root, "Fixture", 4)
        directory = self.root / "Fixture-Taxi"
        (directory / "roadmap.rel").unlink()
        write_csv(directory / "roadmap.rel", ["origin_id", "destination_id"],
                  [{"origin_id": 0, "destination_id": 1}])
        with self.assertRaisesRegex(ValueError, "Highest geography ID"):
            verify_export(directory, self.root / "Trajs_Fixture.txt", "Fixture")


if __name__ == "__main__":
    unittest.main()
