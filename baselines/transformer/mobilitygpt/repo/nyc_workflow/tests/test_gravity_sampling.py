"""Synthetic checks only; no prepared NYC data or full training is required."""

import contextlib
import csv
import io
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler

from datasets.char_dataset import CharDataset
from nyc_workflow.data import GravityTrajectoryDataset
from finetuners.supervised_finetuning import SupervisedFinetuner
from mobilitygpt import trainer_utils
from mobilitygpt.config import get_base_config
from mobilitygpt.model import GPT


def char_dataset(roads, block_size=4):
    text = "\n\n".join(",".join(map(str, trajectory)) for trajectory in roads)
    vocab = sorted({str(road) for trajectory in roads for road in trajectory}, key=int)
    with contextlib.redirect_stdout(io.StringIO()):
        return CharDataset(SimpleNamespace(block_size=block_size), text, vocab)


class WindowTests(unittest.TestCase):
    def test_padding_ignores_only_artificial_targets_and_keeps_terminal_eos(self):
        base = char_dataset([[0, 1]], block_size=6)
        dataset = GravityTrajectoryDataset(base)
        x, y = dataset[0]
        eos = base.stoi[base.EOS_TOKEN]
        self.assertEqual(x.tolist(), [eos, 0, 1, eos, eos, eos])
        self.assertEqual(y.tolist(), [0, 1, eos, -1, -1, -1])
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_exact_size_sequence_has_no_padding(self):
        base = char_dataset([[0, 1, 2]], block_size=4)
        x, y = GravityTrajectoryDataset(base)[0]
        self.assertEqual(x.tolist(), [3, 0, 1, 2])
        self.assertEqual(y.tolist(), [0, 1, 2, 3])

    def test_random_windows_are_local_and_tail_is_reachable(self):
        base = char_dataset([list(range(20)), [20, 20]], block_size=4)
        dataset = GravityTrajectoryDataset(base)
        original = [list(trajectory) for trajectory in base.trajs]
        max_offset = len(base.trajs[0]) - 1 - 4
        with patch("nyc_workflow.data.torch.randint", return_value=torch.tensor([max_offset])):
            x, y = dataset[0]
            self.assertEqual(x[-1].item(), base.stoi["19"])
            self.assertEqual(y[-1].item(), base.stoi[base.EOS_TOKEN])
        offsets_seen = set()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for _ in range(250):
                x, y = dataset[0]
                offsets_seen.add(x[0].item())
                self.assertNotIn(base.stoi["20"], x.tolist() + y.tolist())
                self.assertTrue(torch.equal(x[1:], y[:-1]))
                x_other, y_other = dataset[1]
                self.assertEqual(y_other.tolist(), [base.stoi["20"], base.stoi["20"], dataset.eos_id, -1])
                self.assertEqual(x_other.shape, (4,))
        self.assertIn(base.stoi["16"], offsets_seen)  # last possible offset
        self.assertGreater(len(offsets_seen), 5)
        self.assertEqual(base.trajs, original)
        self.assertEqual(len(dataset), 2)

    def test_fixed_size_batches_and_cpu_model_forward_backward(self):
        torch.set_num_threads(1)
        dataset = GravityTrajectoryDataset(char_dataset([[0, 1], list(range(12))], block_size=8))
        x, y = next(iter(DataLoader(dataset, batch_size=2)))
        config = get_base_config().model
        config.model_type = None
        config.n_layer, config.n_head, config.n_embd = 1, 2, 8
        config.block_size = 8
        config.vocab_size = dataset.get_vocab_size()
        # Exercise unchanged adjacency masking and the model's ignore_index=-1.
        adjacency = torch.ones(config.vocab_size, config.vocab_size)
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(config, adj_matrix=adjacency)
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 8, config.vocab_size))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1))[y.reshape(-1) != -1], y[y != -1]
        )
        torch.testing.assert_close(loss, expected)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        original = Path.cwd()
        self.addCleanup(os.chdir, original)
        os.chdir(temporary.name)
        self.directory = Path("Fixture-Taxi")
        self.directory.mkdir()
        self.gravity_path = self.directory / "Fixture_Taxi_trajectory_train_w_gravity.csv"
        self.source_path = self.directory / "Fixture_Taxi_trajectory_train.csv"
        self.config = get_base_config()
        self.config.data.dataset = "Fixture"
        self.config.system.work_dir = temporary.name
        self.config.training.validation_split = 0
        self.config.training.shuffle_dataset = False
        self.config.training.random_seed = 42

    def fixture(self, weights=(1, 3, 9), parents=None, roads=None, original=False):
        roads = roads or [[i] * 10 for i in range(len(weights))]
        base = char_dataset(roads)
        rows = [{"traj_id": f"trip-{i}", "rid_list": ",".join(map(str, trajectory)), "gravity": weight}
                for i, (trajectory, weight) in enumerate(zip(roads, weights))]
        if parents is not None:
            for row, parent in zip(rows, parents):
                row["parent_traj_id"] = parent
        self.write_rows(self.gravity_path, rows)
        if original:
            self.write_rows(self.source_path, [{k: v for k, v in row.items() if k != "gravity"} for row in rows])
        return base, GravityTrajectoryDataset(base), rows

    def write_rows(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def samplers(self, dataset):
        return trainer_utils.create_gravity_samplers(dataset, self.config)

    def test_weights_select_matching_trajectories_including_last_index(self):
        base, dataset, _ = self.fixture()
        train, validation = self.samplers(dataset)
        self.assertIsInstance(train, WeightedRandomSampler)
        self.assertEqual(len(train.weights), len(dataset))
        self.assertGreater(len(base), len(dataset))
        draws = [index for _ in range(3000) for index in train]
        observed = Counter(draws)
        for index, weight in enumerate((1, 3, 9)):
            self.assertAlmostEqual(observed[index] / len(draws), weight / 13, delta=0.02)
            _, y = dataset[index]
            self.assertTrue(all(token in (index, dataset.eos_id) for token in y.tolist()))
        self.assertEqual(list(validation), [])
        x, y = next(iter(DataLoader(dataset, sampler=train, batch_size=3)))
        self.assertEqual(tuple(x.shape), (3, 4))
        self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))

    def test_equal_positive_weights_remain_sampleable(self):
        _, dataset, _ = self.fixture(weights=(4, 4, 4))
        train, _ = self.samplers(dataset)
        self.assertEqual(train.weights.tolist(), [1, 1, 1])
        self.assertEqual(set(index for _ in range(100) for index in train), {0, 1, 2})

    def test_zero_weights_never_sampled_but_validation_is_uniform(self):
        _, dataset, _ = self.fixture(weights=(0, 2, 0, 4))
        train, _ = self.samplers(dataset)
        self.assertEqual(set(index for _ in range(100) for index in train), {1, 3})
        self.config.training.validation_split = 0.5
        train, validation = self.samplers(dataset)
        self.assertEqual(train.weights.tolist(), [0, 0, 0, 1])
        self.assertIsInstance(validation, SubsetRandomSampler)
        self.assertEqual(set(validation), {0, 1})
        self.assertEqual(len(train), 2)

    def test_group_splits_are_deterministic_and_parent_disjoint(self):
        parents = ["a", "a", "b", "b", "c", "d"]
        _, dataset, _ = self.fixture(weights=(1,) * 6, parents=parents, original=True)
        self.config.training.validation_split = 0.5
        self.config.training.shuffle_dataset = True
        state = np.random.get_state()
        train, validation = self.samplers(dataset)
        train_again, val_again = self.samplers(dataset)
        self.assertEqual(validation.indices, val_again.indices)
        self.assertEqual(list(train), list(train_again))
        self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))
        held_out = set(validation.indices)
        training = set(range(len(dataset))) - held_out
        self.assertFalse({parents[i] for i in held_out} & {parents[i] for i in training})
        self.assertTrue(all(train.weights[i] == 0 for i in held_out))
        self.assertEqual(len(train.weights), 6)
        self.assertEqual(set(validation), held_out)
        self.assertEqual(len(train), len(training))
        self.assertTrue(set(index for _ in range(100) for index in train) <= training)

    def test_original_csv_preserves_parent_grouping_if_gravity_column_is_missing(self):
        _, dataset, rows = self.fixture(weights=(1,) * 4, parents=["a", "a", "b", "b"], original=True)
        for row in rows:
            del row["parent_traj_id"]
        self.write_rows(self.gravity_path, rows)
        self.config.training.validation_split = 0.5
        train, validation = self.samplers(dataset)
        self.assertEqual(validation.indices, [0, 1])
        self.assertEqual(train.weights.tolist(), [0, 0, 1, 1])

    def test_no_parent_column_splits_original_row_indices(self):
        _, dataset, _ = self.fixture(weights=(1,) * 4, original=True)
        self.config.training.validation_split = 0.5
        train, validation = self.samplers(dataset)
        self.assertEqual(validation.indices, [0, 1])
        self.assertEqual(set(index for _ in range(50) for index in train), {2, 3})

    def test_validation_weights_do_not_set_training_scale(self):
        _, dataset, _ = self.fixture(weights=(1e308, 1e-300, 1e-300))
        self.config.training.validation_split = 1 / 3
        train, validation = self.samplers(dataset)
        self.assertEqual(validation.indices, [0])
        self.assertEqual(train.weights.tolist(), [0, 1, 1])

    def test_training_scale_rejects_underflow_of_positive_weights(self):
        _, dataset, _ = self.fixture(weights=(1e308, 1e-300, 1))
        with self.assertRaisesRegex(ValueError, "dynamic range"):
            self.samplers(dataset)

    def test_missing_artifact_errors_with_generation_command(self):
        dataset = GravityTrajectoryDataset(char_dataset([[0, 1]]))
        with patch.object(trainer_utils, "create_random_samplers", side_effect=AssertionError("uniform fallback")):
            with self.assertRaisesRegex(FileNotFoundError, "GravityBasedSampler.*Fixture"):
                trainer_utils.create_samplers(dataset, self.config)

    def test_invalid_or_nonfinite_weights_rejected(self):
        for weight in (-1, "nan", "inf", "-inf", "", "nonsense"):
            with self.subTest(weight=weight):
                _, dataset, _ = self.fixture(weights=(1, weight, 3))
                with self.assertRaisesRegex(ValueError, "gravity.*(row|finite)"):
                    self.samplers(dataset)

    def test_all_zero_training_weight_rejected_even_if_validation_is_positive(self):
        _, dataset, _ = self.fixture(weights=(5, 0, 0))
        self.config.training.validation_split = 1 / 3
        with self.assertRaisesRegex(ValueError, "no positive total training weight"):
            self.samplers(dataset)
        self.config.training.validation_split = 0
        _, dataset, _ = self.fixture(weights=(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "no positive total training weight"):
            self.samplers(dataset)

    def test_count_order_and_road_mismatches_rejected(self):
        for mutation, message in (
            (lambda rows: rows[:-1], "count mismatch"),
            (lambda rows: rows[::-1], "road sequences and row order"),
            (lambda rows: [dict(rows[0], rid_list="999"), *rows[1:]], "differs from the text"),
        ):
            _, dataset, rows = self.fixture()
            self.write_rows(self.gravity_path, mutation(rows))
            with self.assertRaisesRegex(ValueError, message):
                self.samplers(dataset)

    def test_identical_roads_still_require_matching_source_identity(self):
        _, dataset, rows = self.fixture(roads=[[0], [0], [0]], original=True)
        rows[0]["traj_id"], rows[1]["traj_id"] = rows[1]["traj_id"], rows[0]["traj_id"]
        self.write_rows(self.gravity_path, rows)
        with self.assertRaisesRegex(ValueError, "traj_id order/identity mismatch"):
            self.samplers(dataset)

    def test_parent_identity_mismatch_and_partial_parent_column_rejected(self):
        _, dataset, rows = self.fixture(parents=["a", "b", "c"], original=True)
        rows[0]["parent_traj_id"] = "wrong"
        self.write_rows(self.gravity_path, rows)
        with self.assertRaisesRegex(ValueError, "parent_traj_id order/identity mismatch"):
            self.samplers(dataset)
        rows[0]["parent_traj_id"] = ""
        self.source_path.unlink()
        self.write_rows(self.gravity_path, rows)
        with self.assertRaisesRegex(ValueError, "parent_traj_id must be nonempty"):
            self.samplers(dataset)

    def test_adapter_required_and_invalid_validation_fractions_rejected(self):
        base, dataset, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "adapter"):
            self.samplers(base)
        for fraction in (-0.1, 1, float("nan")):
            self.config.training.validation_split = fraction
            with self.assertRaisesRegex(ValueError, "validation_split"):
                self.samplers(dataset)

    def test_pretraining_routes_adapter_only_for_weighted_path(self):
        base, _, _ = self.fixture()
        with patch.object(trainer_utils, "Trainer") as trainer, patch.object(trainer_utils, "save_checkpoint"):
            trainer_utils.train_base_model(MagicMock(), base, self.config)
            kwargs = trainer.call_args.kwargs
            self.assertIsInstance(kwargs["train_dataset"], GravityTrajectoryDataset)
            self.assertIsInstance(kwargs["train_sampler"], WeightedRandomSampler)
            self.assertIs(kwargs["train_dataset"].base_dataset, base)
        self.config.training.gravity_sampling = False
        self.gravity_path.unlink()
        with patch.object(trainer_utils, "Trainer") as trainer, \
                patch.object(trainer_utils, "save_checkpoint"), \
                patch.object(trainer_utils, "load_gravity_data", side_effect=AssertionError("gravity load")), \
                patch.object(trainer_utils, "GravityTrajectoryDataset", side_effect=AssertionError("adapter")):
            trainer_utils.train_base_model(MagicMock(), base, self.config)
            kwargs = trainer.call_args.kwargs
            self.assertIs(kwargs["train_dataset"], base)
            self.assertIsInstance(kwargs["train_sampler"], SubsetRandomSampler)

    def test_supervised_dpo_and_supervised_fallbacks_do_not_use_gravity(self):
        dataset = char_dataset([[0, 1, 2]] * 4)
        for mode in ("supervised", "dpo", "ppo"):
            self.config.training.mode = mode
            with self.subTest(mode=mode), \
                    patch.object(trainer_utils, "GravityTrajectoryDataset", side_effect=AssertionError("adapter")), \
                    patch.object(trainer_utils, "load_gravity_data", side_effect=AssertionError("gravity load")), \
                    patch.object(trainer_utils, "save_checkpoint"), \
                    patch.object(trainer_utils, "GPT", return_value=MagicMock()), \
                    patch.object(trainer_utils, "Agent", return_value=MagicMock()), \
                    patch("finetuners.supervised_finetuning.RewardTrainer") as supervised_trainer, \
                    patch("finetuners.dpo_finetuning.DPOTrainer") as dpo_trainer, \
                    patch.object(trainer_utils, "PPOFinetuner"):
                # Real finetuner constructors and sampler setup run, including
                # supervised fallbacks. Only the backend training loops are mocked.
                if mode == "supervised":
                    trainer_utils.train_supervised(MagicMock(), dataset, self.config)
                elif mode == "dpo":
                    trainer_utils.train_dpo(MagicMock(), dataset, self.config)
                    dpo_trainer.return_value.run.assert_called_once()
                    self.assertIs(dpo_trainer.call_args.kwargs["train_dataset"], dataset)
                else:
                    trainer_utils.train_ppo(MagicMock(), dataset, self.config)
                supervised_trainer.return_value.run.assert_called_once()
                self.assertIs(supervised_trainer.call_args.args[2], dataset)
                train, validation = trainer_utils.create_samplers(dataset, self.config)
                self.assertIsInstance(train, SubsetRandomSampler)
                self.assertIsInstance(validation, SubsetRandomSampler)
                actual = SupervisedFinetuner(self.config, MagicMock(), dataset).create_data_samplers()
                self.assertTrue(all(isinstance(s, SubsetRandomSampler) for s in actual))


if __name__ == "__main__":
    unittest.main()
