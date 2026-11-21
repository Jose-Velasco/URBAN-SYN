"""Regression tests for the two execution fixes, not reward semantics."""

import contextlib
import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from mobilitygpt.config import get_base_config
from mobilitygpt.model import GPT
from mobilitygpt.reward_trainer import RewardTrainer


def training_config():
    config = get_base_config().training
    config.num_workers = 0
    config.batch_size = 1
    return config


def connected_pairs(device="cpu"):
    config = get_base_config().model
    config.model_type = None
    config.n_layer, config.n_head, config.n_embd = 1, 2, 8
    config.block_size, config.vocab_size = 5, 4
    config.embd_pdrop = config.resid_pdrop = config.attn_pdrop = 0.0
    adjacency = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1],
                              [0, 0, 0, 1], [1, 1, 1, 1]],
                             dtype=torch.bool, device=device)
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(config, adj_matrix=adjacency).to(device)
    # Leading EOS, distinct connected routes, and original EOS padding.
    x = torch.tensor([[[3, 0, 1, 2, 3], [3, 1, 2, 3, 3]]])
    y = torch.tensor([[0, 1]])
    return model, TensorDataset(x, y)


class ExtremeLogits(nn.Module):
    def __init__(self):
        super().__init__()
        limit = torch.finfo(torch.float32).max
        self.values = nn.Parameter(torch.tensor([[limit, -limit], [-limit, limit]]))

    def forward(self, x):
        return self.values[:, None, :], None


class SupervisedExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(160)

    def run_connected(self, device):
        model, dataset = connected_pairs(device)
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        captured = []
        handle = model.register_forward_hook(lambda m, a, out: captured.append(out[0].detach()))
        self.addCleanup(handle.remove)
        trainer = RewardTrainer(training_config(), model, dataset)
        trainer.run()
        logits = captured[0]
        chosen, rejected = logits[::2].squeeze(-1), logits[1::2].squeeze(-1)
        # The former expression actually overflows for this connected pair.
        old_loss = -torch.log(torch.sigmoid(chosen - rejected)).mean()
        self.assertFalse(torch.isfinite(old_loss).item())
        expected = -F.logsigmoid(chosen.double() - rejected.double()).mean()
        torch.testing.assert_close(trainer.loss, expected)
        self.assertEqual(tuple(chosen.shape), (1, 5, 4))
        self.assertEqual(trainer.loss.dtype, torch.float64)
        self.assertTrue(torch.isfinite(trainer.loss).item())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(any(not torch.equal(p, before[n]) for n, p in model.named_parameters()))
        self.assertEqual(trainer.iter_num, 1)  # Original one-pass exhaustion.
        self.assertEqual(trainer.max_iters, 3000)
        self.assertEqual(len(captured), 1)  # No validation forwards.

    def test_connected_pairs_with_different_masks_have_finite_gradients(self):
        self.run_connected("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA access required")
    def test_cpu_loader_inputs_are_moved_to_cuda_model(self):
        self.run_connected("cuda")

    def test_cast_before_subtraction_prevents_extreme_float32_overflow(self):
        model = ExtremeLogits()
        raw_difference = model.values[0] - model.values[1]
        self.assertFalse(torch.isfinite(raw_difference).all())
        x = torch.zeros((1, 2, 1), dtype=torch.long)
        trainer = RewardTrainer(training_config(), model,
                                TensorDataset(x, torch.tensor([[0, 1]])))
        trainer.run()
        self.assertTrue(torch.isfinite(trainer.loss))
        self.assertEqual(trainer.loss.dtype, torch.float64)
        self.assertTrue(torch.isfinite(model.values.grad).all())
        self.assertGreater(model.values.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.values).all())

    def test_moderate_logits_preserve_original_mathematical_objective(self):
        model = ExtremeLogits()
        with torch.no_grad():
            model.values.copy_(torch.tensor([[2.0, -3.0], [-1.0, 4.0]]))
        expected = -torch.log(torch.sigmoid(
            model.values[0].double() - model.values[1].double())).mean()
        x = torch.zeros((1, 2, 1), dtype=torch.long)
        trainer = RewardTrainer(training_config(), model,
                                TensorDataset(x, torch.tensor([[0, 1]])))
        trainer.run()
        torch.testing.assert_close(trainer.loss, expected)


if __name__ == "__main__":
    unittest.main()
