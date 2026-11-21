"""CPU-only regression checks for shared adjacency masking and decoding."""

import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from mobilitygpt.config import get_base_config
from mobilitygpt.dpo_trainer import get_log_ps
from mobilitygpt.model import GPT


EOS = 3
ITOS = {0: "0", 1: "1", 2: "2", EOS: "</S>"}


def directed_network():
    # Road 2 is a road dead end, but can still terminate at EOS.
    return torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1],
                         [0, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)


def tiny_model(adjacency=True):
    config = get_base_config().model
    config.model_type = None
    config.n_layer, config.n_head, config.n_embd = 1, 2, 8
    config.block_size, config.vocab_size = 5, 4
    with contextlib.redirect_stdout(io.StringIO()):
        model = GPT(config, adj_matrix=directed_network() if adjacency else None)
    return model.eval()


class ControlledHead(nn.Module):
    def forward(self, hidden):
        # The forbidden road 0 dominates before masking; road 1 is a valid zero.
        return hidden.new_tensor([100.0, 0.0, -5.0, -10.0]).expand(*hidden.shape[:-1], 4)


class AdjacencyMaskingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(42)
        self.model = tiny_model()

    def test_membership_preserves_allowed_values_and_zeroes_forbidden_probability(self):
        idx = torch.tensor([[0, 1, 2], [EOS, 1, 0]])
        # Nonbinary adjacency values still mean membership, not a logit scale.
        self.model.adj_matrix[0, 1] = 2
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits = (torch.arange(24).reshape(2, 3, 4) - 10).to(dtype)
                logits[0, 0, 1], logits[0, 0, 3], logits[0, 1, 2] = 3, -2, 0
                before = logits.clone()
                forbidden = self.model.adj_matrix[idx].eq(0)
                masked = self.model._mask_adjacency_logits(logits, idx)
                self.assertTrue(torch.equal(masked[~forbidden], logits[~forbidden]))
                self.assertTrue(torch.equal(logits, before))
                self.assertTrue(torch.isfinite(masked).all())
                self.assertTrue(torch.all(masked[forbidden] == torch.finfo(dtype).min))
                probabilities = masked.softmax(dim=-1)
                self.assertTrue(torch.all(probabilities[forbidden] == 0))
                self.assertTrue(torch.isfinite(probabilities).all())
                torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)))

    def test_forbidden_logits_have_zero_gradient(self):
        idx = torch.tensor([[0, 1], [2, EOS]])
        logits = torch.randn(2, 2, 4, requires_grad=True)
        masked = self.model._mask_adjacency_logits(logits, idx)
        masked.backward(torch.ones_like(masked))
        expected = self.model.adj_matrix[idx].ne(0).to(logits.dtype)
        self.assertTrue(torch.equal(logits.grad, expected))

    def test_forward_policy_and_reward_slice_share_mask(self):
        idx = torch.tensor([[EOS, 0, 1], [0, 1, 2]])
        raw = []
        handle = self.model.lm_head.register_forward_hook(
            lambda module, inputs, output: raw.append(output.detach().clone())
        )
        self.addCleanup(handle.remove)
        logits, _ = self.model(idx)
        policy_logits, hidden = self.model.policy(idx)
        self.assertTrue(torch.equal(logits, policy_logits))
        self.assertTrue(torch.equal(logits, self.model._mask_adjacency_logits(raw[0], idx)))
        self.assertEqual(tuple(hidden.shape), (2, 3, 8))
        self.model.reward_model = True
        reward_logits, _ = self.model(idx)
        self.assertTrue(torch.equal(reward_logits, logits[:, -1, :]))

    def test_connected_targets_and_ignored_padding_have_finite_loss_and_gradients(self):
        idx = torch.tensor([[EOS, 0, 1, 2], [EOS, 1, 2, EOS]])
        targets = torch.tensor([[0, 1, 2, EOS], [1, 2, EOS, -1]])
        logits, loss = self.model(idx, targets)
        self.assertTrue(torch.isfinite(loss))
        active = targets.ne(-1)
        expected = F.cross_entropy(logits[active], targets[active])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_dpo_zero_mask_on_forbidden_gather_stays_finite(self):
        idx = torch.tensor([[EOS, 0, 2, EOS]])  # 0 -> 2 is forbidden, but unscored.
        completion_mask = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        logits, _ = self.model(idx)
        scores = get_log_ps(logits, idx, completion_mask)
        expected = F.log_softmax(logits, dim=-1)[0, 0, 0]
        torch.testing.assert_close(scores[0], expected)  # Road 2 -> EOS has log p = 0.
        self.assertTrue(torch.isfinite(scores).all())
        (-scores.sum()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all()
                            for p in self.model.parameters() if p.grad is not None))

    def test_dead_end_terminates_at_eos(self):
        self.model.lm_head = ControlledHead()
        prompt = torch.tensor([[EOS, 2]])
        for sampled in (False, True):
            with self.subTest(sampled=sampled):
                result = self.model.generate_test(
                    prompt, ITOS, "</S>", do_sample=sampled, max_token=6
                )
                self.assertTrue(torch.equal(result, prompt))

    def test_allowed_zero_logit_remains_a_generation_candidate(self):
        self.model.lm_head = ControlledHead()
        result = self.model.generate_test(
            torch.tensor([[EOS, 0]]), ITOS, "</S>", max_token=6
        )
        self.assertEqual(result.tolist(), [[EOS, 0, 1, 2]])

    def test_decoding_combinations_remain_connected_and_probabilities_normalized(self):
        self.model.lm_head = ControlledHead()
        original_softmax = F.softmax
        for temperature in (0.7, 1.0, 1.3):
            for top_k in (None, 2):
                for sampled in (False, True):
                    with self.subTest(temperature=temperature, top_k=top_k, sampled=sampled):
                        contexts, probabilities = [], []
                        handle = self.model.register_forward_pre_hook(
                            lambda module, inputs: contexts.append(inputs[0].clone())
                        )

                        def record_softmax(values, *args, **kwargs):
                            result = original_softmax(values, *args, **kwargs)
                            if values.ndim == 2:  # Generation, not attention softmax.
                                probabilities.append(result.clone())
                            return result

                        try:
                            with patch("mobilitygpt.model.F.softmax", side_effect=record_softmax):
                                torch.manual_seed(7)
                                result = self.model.generate_test(
                                    torch.tensor([[EOS, 0]]), ITOS, "</S>",
                                    temperature=temperature, top_k=top_k,
                                    do_sample=sampled, max_token=6,
                                )
                        finally:
                            handle.remove()
                        roads = result[0].tolist()
                        self.assertTrue(all(self.model.adj_matrix[a, b] != 0
                                            for a, b in zip(roads, roads[1:])))
                        self.assertEqual(len(contexts), len(probabilities))
                        self.assertTrue(probabilities)
                        for context, probability in zip(contexts, probabilities):
                            forbidden = self.model.adj_matrix[context[:, -1]].eq(0)
                            self.assertTrue(torch.isfinite(probability).all())
                            self.assertTrue(torch.all(probability[forbidden] == 0))
                            torch.testing.assert_close(probability.sum(-1), torch.ones(1))
                        self.assertGreater(probabilities[0][0, 1].item(), 0)

    def test_existing_length_cap_is_preserved(self):
        self.model.lm_head = ControlledHead()
        result = self.model.generate_test(
            torch.tensor([[EOS, 0]]), ITOS, "</S>", max_token=3
        )
        self.assertEqual(result.tolist(), [[EOS, 0, 1]])

    def test_no_adjacency_is_identity(self):
        self.model.adj_matrix = None
        idx = torch.tensor([[EOS, 0, 1]])
        raw = torch.randn(1, 3, 4)
        self.assertIs(self.model._mask_adjacency_logits(raw, idx), raw)
        captured = []
        handle = self.model.lm_head.register_forward_hook(
            lambda module, inputs, output: captured.append(output.detach().clone())
        )
        try:
            logits, _ = self.model(idx)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(logits, captured[0]))

    def test_state_dict_schema_and_strict_loading_are_preserved(self):
        # Original one-layer architecture schema, including existing LoRA tensors.
        expected = {
            "transformer.wte.weight": (4, 8), "transformer.wpe.weight": (5, 8),
            "transformer.ln_f.weight": (8,), "transformer.ln_f.bias": (8,),
            "lm_head.weight": (4, 8), "transformer.h.0.attn.bias": (1, 1, 5, 5),
            "transformer.h.0.attn.c_attn.weight": (24, 8),
            "transformer.h.0.attn.c_attn.lora_A": (8, 8),
            "transformer.h.0.attn.c_attn.lora_B": (24, 8),
            "transformer.h.0.attn.c_proj.weight": (8, 8),
            "transformer.h.0.attn.c_proj.lora_A": (8, 8),
            "transformer.h.0.attn.c_proj.lora_B": (8, 8),
            "transformer.h.0.ln_1.weight": (8,), "transformer.h.0.ln_1.bias": (8,),
            "transformer.h.0.ln_2.weight": (8,), "transformer.h.0.ln_2.bias": (8,),
            "transformer.h.0.mlp.c_fc.weight": (32, 8), "transformer.h.0.mlp.c_fc.bias": (32,),
            "transformer.h.0.mlp.c_proj.weight": (8, 32), "transformer.h.0.mlp.c_proj.bias": (8,),
        }
        state = self.model.state_dict()
        self.assertEqual({key: tuple(value.shape) for key, value in state.items()}, expected)
        restored = tiny_model(adjacency=False)
        status = restored.load_state_dict(state, strict=True)
        self.assertEqual(status.missing_keys, [])
        self.assertEqual(status.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
