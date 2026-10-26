"""Synthetic verification of the standalone checkpoint evaluation protocol."""

import copy
import unittest

import torch

from nyc_workflow.paired import (
    CHECKPOINTS, SEEDS, generate_route, route_distance, route_quality,
    select_validation, stable_seed, verify_pairing,
)


class ConstantModel:
    block_size = 81
    adj_matrix = torch.ones(2, 2)

    def __init__(self, eos=False, invalid=False):
        self.eos, self.invalid = eos, invalid

    def __call__(self, idx):
        values = [-1000.0, 0.0] if self.eos else [0.0, -1000.0]
        if self.invalid:
            values = [float("nan"), 0.0]
        return torch.tensor(values).expand(1, idx.shape[1], 2), None


class CheckpointComparisonTests(unittest.TestCase):
    def test_distance_counts_each_revisit_in_meters(self):
        lengths = {0: 10.0, 1: 20.0}
        self.assertEqual(route_distance([0, 1, 0], lengths), 40)
        quality = route_quality([0, 1, 0], lengths, {(0, 1), (1, 0)})
        self.assertEqual(quality["distance_m"], 40)
        self.assertEqual(quality["invalid_transitions"], 0)
        self.assertTrue(quality["has_repeated_road"])
        self.assertAlmostEqual(quality["unique_road_fraction"], 2 / 3)

    def test_selection_is_distinct_deterministic_validation_only_and_parent_disjoint(self):
        rows = [{"traj_id": str(i), "parent_traj_id": str(i // 2), "rid_list": "0,1,2,1"}
                for i in range(8)]
        split = {"train_indices": [0, 1, 2, 3], "validation_indices": [4, 5, 6, 7],
                 "train_parents": ["0", "1"], "validation_parents": ["2", "3"]}
        selected = select_validation(rows, split, count=3)
        self.assertEqual(selected, select_validation(rows, split, count=3))
        self.assertEqual(len({r["trajectory_id"] for r in selected}), 3)
        self.assertTrue({r["trajectory_index"] for r in selected} <= {4, 5, 6, 7})
        self.assertTrue({r["parent_id"] for r in selected} <= {"2", "3"})
        split["train_indices"] = [0, 1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_validation(rows, split, count=3)

    def test_seed_is_stable_and_depends_on_all_three_fields(self):
        value = stable_seed(42, "trip:0", 1)
        self.assertEqual(value, stable_seed(42, "trip:0", 1))
        self.assertNotEqual(value, stable_seed(43, "trip:0", 1))
        self.assertNotEqual(value, stable_seed(42, "trip:1", 1))
        self.assertNotEqual(value, stable_seed(42, "trip:0", 3))

    def test_eos_and_existing_cap_priority(self):
        prompt = torch.tensor([[1, 0]])
        roads, reason, steps, _ = generate_route(ConstantModel(), prompt, 1, 42)
        self.assertEqual((len(roads), reason, steps), (80, "length_cap", 80))
        full = torch.tensor([[1] + [0] * 80])
        roads, reason, steps, _ = generate_route(ConstantModel(eos=True), full, 1, 42)
        self.assertEqual((len(roads), reason, steps), (80, "eos", 1))
        roads, reason, _, _ = generate_route(ConstantModel(eos=True), prompt, 1, 42)
        self.assertEqual((roads, reason), ([0], "eos"))

    def test_nonfinite_probabilities_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            generate_route(ConstantModel(invalid=True), torch.tensor([[1, 0]]), 1, 42)

    def test_pairing_detects_dropped_samples_or_mismatched_seed_and_prompt(self):
        selected = [{"trajectory_id": "v0", "parent_id": "p0", "reference_roads": [0, 1, 2, 1],
                     "eligible_prompt_lengths": [1, 3]}]
        records = []
        for checkpoint in CHECKPOINTS:
            for prompt in (1, 3):
                for seed in SEEDS:
                    roads = selected[0]["reference_roads"][:prompt]
                    records.append({"checkpoint": checkpoint, "trajectory_id": "v0", "parent_id": "p0",
                                    "prompt_length": prompt, "prompt_roads": roads,
                                    "seed": seed, "derived_seed": stable_seed(seed, "v0", prompt),
                                    "roads": roads, "road_count": len(roads),
                                    "termination": "eos", "immediate_stop": True})
        self.assertEqual(verify_pairing(records, selected)["actual_samples"], 12)
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify_pairing(records[:-1], selected)
        bad = copy.deepcopy(records)
        bad[0]["derived_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Seed mismatch"):
            verify_pairing(bad, selected)
        bad = copy.deepcopy(records)
        bad[0]["prompt_roads"] = [2]
        with self.assertRaisesRegex(ValueError, "Prompt mismatch"):
            verify_pairing(bad, selected)


if __name__ == "__main__":
    unittest.main()
