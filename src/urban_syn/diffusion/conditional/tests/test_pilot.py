"""Focused alignment and conditioning regression checks; no dataset training."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from ..conversion import encode_rates
from ..metadata import build_sidecars, iter_batches, load_aligned_split, replay_alignment
from ..pilot import make_model


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        (self.dataset / "extra_file").mkdir(parents=True)
        (self.dataset / "extra_file/raw2new_rid.json").write_text('{"10": 1, "11": 2}')
        (self.dataset / "extra_file/raw_rn_dict.json").write_text('{"10": {}, "11": {}}')
        first_day = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
        records = []
        for source_id, uid, tid, piece, day, length in [
            (1, "1", "A", 0, 0, 3), (2, "1", "A", 1, 1, 3),
            (3, "2", "B", 0, 0, 3), (4, "3", "C", 0, 0, 2),
        ]:
            start = first_day + day * 86400
            records.append(dict(id=source_id, uid=uid, tid=tid, move_id="0", piece=piece,
                                label="bus", start_timestamp=start,
                                end_timestamp=start + 15 * (length - 1), num_points=length))
        self.trip_map = self.root / "trip_map.csv"
        pd.DataFrame(records).to_csv(self.trip_map, index=False)
        self.fmm = self.root / "fmm.csv"
        pd.DataFrame([
            dict(id=r["id"], opath=",".join(["10"] * r["num_points"]),
                 offset=",".join(map(str, range(r["num_points"]))),
                 length=",".join(["10"] * r["num_points"])) for r in records
        ] + [dict(id=0, opath="", offset="", length="")]).to_csv(self.fmm, sep=";", index=False)
        self.test = self.root / "test.csv"
        pd.DataFrame([dict(uid="3", tid="C")]).to_csv(self.test, index=False)
        self.weather = self.root / "weather.parquet"
        pd.DataFrame([
            dict(uid="1", tid="A", DATE="2020-01-01", DESCRIPTION="sunny"),
            dict(uid="1", tid="A", DATE="2020-01-02", DESCRIPTION="heavy rain"),
            dict(uid="3", tid="C", DATE="2020-01-01", DESCRIPTION="light rain"),
        ]).to_parquet(self.weather, index=False)
        rates3 = encode_rates([10] * 3, [0, 1, 2], [10] * 3)
        rates2 = encode_rates([10] * 2, [0, 1], [10] * 2)
        for split, ids, rates in [
            ("gen_train", {3: [[1] * 3] * 3}, {3: [rates3] * 3}),
            ("gen_test", {2: [[1] * 2]}, {2: [rates2]}),
            ("gen_all", {2: [[1] * 2], 3: [[1] * 3] * 3}, {2: [rates2], 3: [rates3] * 3}),
        ]:
            folder = self.dataset / split
            folder.mkdir()
            for name, value in [("eid_seqs.bin", ids), ("rate_seqs.bin", rates)]:
                with (folder / name).open("wb") as stream:
                    pickle.dump(value, stream)

    def build(self):
        return build_sidecars(self.dataset, self.trip_map, self.fmm, self.test,
                              self.weather, self.root / "metadata")

    def test_daily_join_unknown_and_training_only_vocabulary(self):
        manifest = self.build()
        self.assertNotIn("light rain", manifest["vocabulary"])
        _, _, codes, rows, _ = load_aligned_split(self.dataset, self.root / "metadata")
        self.assertEqual(rows.fmm_id.tolist(), [1, 2, 3])
        self.assertEqual(codes[3], [manifest["vocabulary"]["sunny"],
                                   manifest["vocabulary"]["heavy rain"], 0])
        self.assertEqual(manifest["coverage"]["gen_train"]["status_counts"]["missing_source"], 1)
        self.assertEqual(manifest["coverage"]["gen_test"]["status_counts"]["unseen_training_category"], 1)
        self.assertEqual(manifest["skipped_fmm_rows"], {"empty_opath": 1})

    def test_replay_rejects_changed_trajectory(self):
        target = self.dataset / "gen_train/eid_seqs.bin"
        with target.open("wb") as stream:
            pickle.dump({3: [[2] * 3, [1] * 3, [1] * 3]}, stream)
        with self.assertRaisesRegex(ValueError, "exact conversion replay failed"):
            replay_alignment(self.dataset, self.trip_map, self.fmm, self.test)

    def test_duplicate_weather_keys_are_rejected(self):
        weather = pd.read_parquet(self.weather)
        pd.concat([weather, weather.iloc[:1]]).to_parquet(self.weather, index=False)
        with self.assertRaisesRegex(ValueError, "not unique"):
            self.build()

    def test_batch_shuffle_preserves_alignment_and_remainders(self):
        ids = {2: [[11, 11], [12, 12], [13, 13]], 3: [[14] * 3]}
        rates = {2: [[.11] * 2, [.12] * 2, [.13] * 2], 3: [[.14] * 3]}
        codes = {2: [11, 12, 13], 3: [14]}
        batches = list(iter_batches(ids, rates, codes, 2, seed=4))
        self.assertEqual(sorted(len(batch[0]) for batch in batches), [1, 1, 2])
        seen = set()
        for eid_batch, rate_batch, weather_batch in batches:
            for eid, rate, weather in zip(eid_batch, rate_batch, weather_batch):
                self.assertEqual(eid[0], weather)
                self.assertAlmostEqual(rate[0], weather / 100)
                seen.add(weather)
        self.assertEqual(seen, {11, 12, 13, 14})
        with self.assertRaises(ValueError):
            list(iter_batches(ids, rates, {2: [11], 3: [14]}, 2))


class ModelTests(unittest.TestCase):
    def test_dummy_forward_backward_generation_and_condition_shuffle(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = make_model(dict(channels=8, steps=2, pre_dim=4, residual_layers=1,
                                weather_categories=3, condition_dim=8,
                                beta_start=1e-4, beta_end=.02))
        embeddings = torch.randn(8, 4)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        weather = torch.tensor([1, 2])
        graph = torch.empty(0, dtype=torch.long)
        losses = model(graph, embeddings, ids, torch.zeros(2, 3), weather)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        sum(losses).backward()
        gradient = model.diff_model.weather_embedding.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum().item(), 0)
        model.eval()
        x = torch.randn(2, 3, 5)
        t = torch.ones(2, 1, dtype=torch.long)
        self.assertFalse(torch.allclose(model.diff_model(x, t, weather),
                                        model.diff_model(x, t, weather.flip(0))))
        for invalid, exception in [(weather[:1], ValueError), (weather.float(), TypeError),
                                   (torch.tensor([-1, 2]), ValueError)]:
            with self.assertRaises(exception):
                model.diff_model(x, t, invalid)
        generated_ids, generated_rates = model.generate_data(
            graph, embeddings, 2, 3, 4, weather, generator=torch.Generator().manual_seed(0))
        self.assertEqual(generated_ids.shape, (2, 3))
        self.assertEqual(generated_rates.shape, (2, 3))
        self.assertTrue(torch.isfinite(generated_rates).all())


if __name__ == "__main__":
    unittest.main()
