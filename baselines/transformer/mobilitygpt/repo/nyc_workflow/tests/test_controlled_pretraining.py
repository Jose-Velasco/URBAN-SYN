"""Checks for leakage prevention and deterministic held-out evaluation."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.char_dataset import CharDataset
from nyc_workflow.data import (
    DeterministicTrajectoryWindows, GravityTrajectoryDataset, split_trajectory_indices,
)
from mobilitygpt.config import get_base_config
from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from preprocessing.gravity_based_sampling import GravityBasedSampler


class ControlledPretrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            base = CharDataset(SimpleNamespace(block_size=4),
                               '0,1,2,3,4,5,6,7\n\n0,1', list(map(str, range(8))))
        self.dataset = GravityTrajectoryDataset(base)

    def test_deterministic_windows_cover_each_target_once(self):
        windows = DeterministicTrajectoryWindows(self.dataset, [1, 0])
        before = torch.random.get_rng_state()
        targets = [y[y != -1].tolist() for _, y in windows]
        expected = [sequence[1:].tolist() for sequence in self.dataset.trajectories]
        self.assertEqual(sum(targets[:3], []), expected[0])
        self.assertEqual(targets[3], expected[1])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertEqual([y.tolist() for _, y in windows], [y.tolist() for _, y in windows])

    def test_token_mean_evaluation_preserves_mode_and_rng(self):
        config = get_base_config()
        config.training.device = 'cpu'
        config.training.batch_size = 2
        config.model.model_type = None
        config.model.n_layer, config.model.n_head, config.model.n_embd = 1, 2, 8
        config.model.block_size, config.model.vocab_size = 4, 9
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(config.model)
            held_out = DeterministicTrajectoryWindows(self.dataset, [0, 1])
            trainer = Trainer(config.training, model, self.dataset, [0], [1], val_dataset=held_out)
        state = torch.random.get_rng_state()
        first = trainer.evaluate()
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(first, trainer.evaluate())
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in DataLoader(held_out, batch_size=2):
                logits, _ = model(x, y)
                losses.extend(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 9), y.reshape(-1), ignore_index=-1,
                    reduction='none')[y.reshape(-1) != -1].tolist())
        self.assertAlmostEqual(first, sum(losses) / len(losses), places=6)
        self.assertEqual(trainer.val_tokens, 12)

    def test_gravity_counts_exclude_validation_parents(self):
        # Exercise process itself: fitted counts must omit both held-out siblings,
        # while output weights are still calculated for every original row.
        frame = pd.DataFrame({'parent_traj_id': ['a', 'a', 'b', 'c'],
                              'rid_list': ['0,1', '1,0', '0,1', '1,0']})
        sampler = GravityBasedSampler('Fixture', validation_split=0.5, shuffle=False)
        roads = pd.DataFrame({'coordinates': ['[[0,0],[1,1]]', '[[1,1],[2,2]]']})
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            sampler.data_dir = directory
            with patch('preprocessing.gravity_based_sampling.pd.read_csv', side_effect=[roads, frame]), \
                 patch.object(sampler, '_convert_gps_to_2d_grid', return_value=(range(2), {0: [0], 1: [1]}, {0: 0, 1: 1})), \
                 patch.object(sampler, '_calculate_region_counts', return_value=np.ones((1, 2))) as counts, \
                 patch.object(sampler, '_calculate_gravity_matrix', return_value=np.ones((2, 2))), \
                 patch('preprocessing.gravity_based_sampling.Path.read_bytes', return_value=b'fixture'):
                sampler.process()
            self.assertEqual(counts.call_args.args[0].parent_traj_id.tolist(), ['b', 'c'])
            output = pd.read_csv(f'{directory}/Fixture_Taxi_trajectory_train_w_gravity.csv')
            self.assertEqual(len(output), 4)
            self.assertEqual(output.parent_traj_id.tolist(), ['a', 'a', 'b', 'c'])

    def test_parent_split_shared_and_disjoint(self):
        parents = ['a', 'a', 'b', 'c', 'd', 'd']
        train, val = split_trajectory_indices(parents, 0.5)
        self.assertEqual((train, val), split_trajectory_indices(parents, 0.5))
        self.assertFalse({parents[i] for i in train} & {parents[i] for i in val})

    def test_trainer_updates_full_backbone_and_rejects_nonfinite_loss(self):
        config = get_base_config()
        config.training.device = 'cpu'
        config.training.batch_size = 2
        config.training.num_workers = 0
        config.training.max_iters = 2
        config.training.learning_rate = 1e-5
        config.model.model_type = None
        config.model.n_layer, config.model.n_head, config.model.n_embd = 1, 2, 8
        config.model.block_size, config.model.vocab_size = 4, 9
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(config.model)
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            old = model.transformer.h[0].attn.c_attn.weight.detach().clone()
            trainer = Trainer(config.training, model, self.dataset, [0, 1], [])
            with patch('mobilitygpt.trainer.ModuleValidator.fix', side_effect=AssertionError('must not modify architecture')):
                trainer.run()
            self.assertEqual(trainer.iter_num, 2)
            self.assertFalse(torch.equal(old, model.transformer.h[0].attn.c_attn.weight))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            with patch.object(model, 'forward', return_value=(None, torch.tensor(float('nan')))):
                with self.assertRaisesRegex(FloatingPointError, 'Nonfinite loss'):
                    trainer.run()


if __name__ == '__main__':
    unittest.main()
