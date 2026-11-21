"""Small CPU checks for boolean adjacency and expanded-run compatibility."""

import contextlib
import copy
import io
import unittest

import torch

from datasets.data_utils import od_pair_to_adjacency_matrix
from mobilitygpt.model import GPT
from nyc_workflow.pretraining import build_config, check_short_compatibility, memory_budget, parse_args as _parse_args


def parse_args(arguments):
    return _parse_args(['--data-root', '/tmp/nyc-prepared-fixture', *arguments])


class NYCAllTrainingTests(unittest.TestCase):
    def test_boolean_adjacency_matches_previous_numeric_values_and_eos(self):
        edges = [[0, 1], [1, 2], [3, 0], [0, 1]]
        old = torch.zeros((4, 4), dtype=int)
        for a, b in edges:
            old[a, b] = 1
        old = torch.cat((old, torch.ones(1, 4)), dim=0)
        old = torch.cat((old, torch.ones(5, 1)), dim=1)
        actual = od_pair_to_adjacency_matrix(edges, 'cpu')
        self.assertEqual(actual.dtype, torch.bool)
        self.assertEqual(tuple(actual.shape), (5, 5))
        self.assertTrue(torch.equal(actual, old.bool()))
        self.assertEqual(actual.numel() * actual.element_size(), 25)
        self.assertEqual(actual[2].tolist(), [False, False, False, False, True])

    def test_boolean_and_numeric_masks_probabilities_losses_and_gradients_match(self):
        adjacency = od_pair_to_adjacency_matrix([[0, 1], [1, 2]], 'cpu')
        config = build_config(parse_args([])).model
        config.model_type = None
        config.n_layer, config.n_head, config.n_embd = 1, 2, 8
        config.block_size, config.vocab_size = 4, 4
        with contextlib.redirect_stdout(io.StringIO()):
            model = GPT(config, adj_matrix=adjacency).eval()
        idx = torch.tensor([[3, 0, 1, 2]])
        target = torch.tensor([[0, 1, 2, 3]])
        logits, loss = model(idx, target)
        loss.backward()
        grads = {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
        model.adj_matrix = adjacency.float()
        numeric_logits, numeric_loss = model(idx, target)
        numeric_loss.backward()
        self.assertTrue(torch.equal(logits, numeric_logits))
        self.assertTrue(torch.equal(logits.softmax(-1), numeric_logits.softmax(-1)))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(loss, numeric_loss))
        for key, parameter in model.named_parameters():
            if key in grads:
                self.assertTrue(torch.equal(grads[key], parameter.grad))
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_dataset_default_selection_and_profiling_settings(self):
        default = build_config(parse_args([]))
        expanded = build_config(parse_args(['--dataset', 'NYCAll', '--profile-only', '--batch-size', '32']))
        self.assertEqual(default.data.dataset, 'NYC20K')
        self.assertEqual(expanded.data.dataset, 'NYCAll')
        self.assertEqual(expanded.training.max_iters, 20)
        self.assertEqual(expanded.training.batch_size, 32)
        self.assertEqual(expanded.training.learning_rate, 1e-5)
        self.assertEqual(expanded.data.block_size, 81)
        self.assertIsNone(expanded.model.load_path)
        self.assertTrue(expanded.model.use_adjacency)

    def test_short_gate_rejects_dataset_model_network_and_training_mismatches(self):
        config = build_config(parse_args(['--dataset', 'NYCAll']))
        diagnostics = {k: 'hash' for k in ('gravity_sha256', 'source_sha256', 'geography_sha256', 'relations_sha256', 'text_sha256')}
        import json
        previous = json.loads(json.dumps(config.to_dict()))
        short = {'passed': True, 'steps': 50, 'memory_budget_passed': True, 'gravity_diagnostics': diagnostics}
        check_short_compatibility(config, diagnostics, short, previous)
        for section, key, value in (('data', 'dataset', 'NYC20K'), ('model', 'n_embd', 128),
                                    ('training', 'batch_size', 64)):
            bad = copy.deepcopy(previous)
            bad[section][key] = value
            with self.assertRaisesRegex(ValueError, 'differs|differ'):
                check_short_compatibility(config, diagnostics, short, bad)
        for key in diagnostics:
            bad = copy.deepcopy(short)
            bad['gravity_diagnostics'][key] = 'other'
            with self.assertRaisesRegex(ValueError, 'data differs'):
                check_short_compatibility(config, diagnostics, bad, previous)
        with self.assertRaisesRegex(ValueError, 'memory budget'):
            check_short_compatibility(config, diagnostics, dict(short, memory_budget_passed=False), previous)

    def test_memory_budget_reserves_headroom(self):
        gib = 1024**3
        self.assertEqual(memory_budget(10 * gib, 10 * gib), int(8.5 * gib))
        self.assertEqual(memory_budget(7 * gib, 10 * gib), int(6.5 * gib))


if __name__ == '__main__':
    unittest.main()
