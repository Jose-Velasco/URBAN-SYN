"""Regression fixtures for supported DPO construction, freezing and milestone saves."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from datasets.data_utils import create_dpo_dataset
from nyc_workflow.dpo import config_for, training
from mobilitygpt.config import get_base_config
from mobilitygpt.dpo_trainer import (KEY_PCT, KEY_PRT, KEY_CLM, KEY_RLM,
                                     compute_loss, get_padded_batch)
from mobilitygpt.model import GPT, get_lora_model


def tiny_pair():
    return {KEY_PCT: torch.tensor([3, 0, 1, 3, 0, 1, 2, 3]),
            KEY_PRT: torch.tensor([3, 0, 1, 3, 0, 1, 2]),
            KEY_CLM: torch.tensor([0., 0., 0., 1., 1., 1., 1., 1.]),
            KEY_RLM: torch.tensor([0., 0., 0., 1., 1., 1., 1.])}


class DPOStageTests(unittest.TestCase):
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

    def models(self):
        config = get_base_config()
        model_config = config.model
        model_config.model_type = None
        model_config.n_layer, model_config.n_head, model_config.n_embd = 1, 2, 8
        model_config.vocab_size, model_config.block_size = 4, 8
        model_config.embd_pdrop = model_config.resid_pdrop = model_config.attn_pdrop = 0.0
        config.system.device = 'cpu'
        config.training.batch_size = 1
        config.training.learning_rate = 1e-5
        config.data.max_length = 8
        adjacency = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1],
                                  [0, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.bool)
        with contextlib.redirect_stdout(io.StringIO()):
            policy = get_lora_model(GPT(model_config, adj_matrix=adjacency))
        reference = copy.deepcopy(policy).requires_grad_(False).eval()
        reference.adj_matrix = policy.adj_matrix
        return config, policy, reference

    def test_original_function_duplicates_prompt_and_scores_inner_copy(self):
        config = get_base_config()
        config.system.device = 'cpu'
        config.training.num_samples, config.training.prompt_size = 1, 4
        eos = '</S>'
        dataset = SimpleNamespace(trajs=[[eos, '0', '1', '2', eos]],
                                  stoi={eos: 3, '0': 0, '1': 1, '2': 2},
                                  itos={0: '0', 1: '1', 2: '2', 3: eos}, EOS_TOKEN=eos)

        class ImmediateStop:
            def generate_test(self, x, *args, **kwargs):
                self.kwargs = kwargs
                return x

        model = ImmediateStop()
        with tempfile.TemporaryDirectory(prefix='dpo_generator_') as temporary:
            config.system.work_dir = temporary
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                create_dpo_dataset(model, dataset, config)
            item = torch.load(Path(temporary) / 'preference_dataset_dpo', weights_only=True)[0]
        self.assertEqual(item[KEY_PCT].tolist(), [3, 0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(item[KEY_PRT].tolist(), [3, 0, 1, 2, 3, 0, 1, 2])
        self.assertEqual(item[KEY_CLM].tolist(), [0.] * 4 + [1.] * 5)
        self.assertEqual(item[KEY_RLM].tolist(), [0.] * 4 + [1.] * 4)
        self.assertEqual(model.kwargs, {'temperature': 1.0, 'do_sample': True, 'top_k': None})

    def test_padding_is_original_road_one_with_zero_completion_mask(self):
        batch = get_padded_batch([tiny_pair()], 8)
        self.assertEqual(batch[1][0, -1].item(), 1)
        self.assertEqual(batch[3][0, -1].item(), 0)
        self.assertTrue(all(t.device.type == 'cpu' for t in batch))

    def test_original_loss_with_forbidden_zero_masked_padding_is_finite(self):
        _, policy, reference = self.models()
        batch = get_padded_batch([tiny_pair()], 8)
        loss, _, _ = compute_loss(policy, reference, batch, 0.1, 'cpu')
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, torch.tensor(2.).log())
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_short_run_freezes_reference_and_bypasses_supervised_fallback(self):
        config, policy, reference = self.models()
        result = {}
        with tempfile.TemporaryDirectory(prefix='dpo_short_run_') as temporary:
            with patch('finetuners.supervised_finetuning.SupervisedFinetuner.train', side_effect=AssertionError('Forbidden supervised fallback')), contextlib.redirect_stdout(io.StringIO()):
                training(policy, reference, config, [tiny_pair()], Path(temporary), result)
            self.assertTrue((Path(temporary) / 'model_dpo.pt').is_file())
            steps = [json.loads(line) for line in (Path(temporary) / 'steps.jsonl').read_text().splitlines()]
        self.assertEqual(result['updates_completed'], 50)
        self.assertTrue(result['reference_unchanged'])
        self.assertTrue(result['reference_gradients_absent'])
        self.assertTrue(result['changed_parameter_names'])
        self.assertEqual(result['frozen_parameter_changes'], [])
        self.assertEqual(len(steps), 50)
        self.assertTrue(all(step['gradients_finite'] for step in steps))

    def test_launch_settings_are_conservative_and_not_supervised(self):
        config = config_for(Path('/tmp/dpo_config_fixture'), pretrained=Path('/tmp/pretrain/model_best.pt'), data_root=Path('/tmp/prepared'))
        self.assertEqual(config.training.mode, 'dpo')
        self.assertEqual(config.training.learning_rate, 1e-5)
        self.assertEqual(config.training.max_iters, 50)
        self.assertTrue(config.model.use_adjacency)
        self.assertTrue(config.model.use_lora)
        self.assertEqual(config.data.dataset, 'NYCAll')

    def test_intermediate_settings_preserve_training_hyperparameters(self):
        config = config_for(Path('/tmp/dpo_intermediate_fixture'), samples=1000, updates=250, pretrained=Path('/tmp/pretrain/model_best.pt'), data_root=Path('/tmp/prepared'))
        self.assertEqual(config.training.num_samples, 1000)
        self.assertEqual(config.training.max_iters, 250)
        self.assertEqual(config.training.learning_rate, 1e-5)
        self.assertEqual(config.training.batch_size, 8)
        self.assertTrue(config.model.use_lora)

    def test_milestones_capture_real_updates_and_strict_reload(self):
        config, policy, reference = self.models()
        result = {}
        with tempfile.TemporaryDirectory(prefix='dpo_milestones_') as temporary:
            directory = Path(temporary)
            with contextlib.redirect_stdout(io.StringIO()):
                training(policy, reference, config, [tiny_pair()], directory, result,
                         updates=4, checkpoint_updates=(1, 2, 4))
            self.assertEqual(result['updates_completed'], 4)
            self.assertEqual(set(result['milestone_checkpoints']), {'1', '2', '4'})
            states = [torch.load(directory / f'model_step_{i:06d}.pt', weights_only=True)
                      for i in (1, 2, 4)]
            self.assertTrue(any(not torch.equal(states[0][key], states[1][key]) for key in states[0]))
            self.assertTrue(result['milestone_checkpoints']['4']['matches_final_model'])
            self.assertTrue(all(meta['reference_unchanged_at_save'] for meta in result['milestone_checkpoints'].values()))
            self.assertTrue(all(meta['strict_cpu_loading'] == '<All keys matched successfully>'
                                for meta in result['milestone_checkpoints'].values()))


if __name__ == '__main__':
    unittest.main()
