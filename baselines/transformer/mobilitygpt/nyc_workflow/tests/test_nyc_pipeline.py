"""Orchestration contracts: isolation, evidence validation, and restart refusals."""

import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from nyc_workflow import pipeline
from datasets.data_utils import load_trajectory_data
from mobilitygpt.config import get_base_config


class PipelineTests(unittest.TestCase):
    def test_locked_recipe_and_no_historical_handoffs(self):
        config = pipeline.configuration('/tmp/raw')
        self.assertEqual(config['pretraining']['updates'], 3000)
        self.assertEqual(config['pretraining']['batch_size'], 64)
        self.assertEqual(config['dpo']['preferences'], 1000)
        self.assertEqual(config['dpo']['milestones'], [50, 100, 250])
        self.assertEqual(config['dpo']['selected_update'], 100)
        self.assertEqual(config['evaluation']['seeds'], [42, 43, 44])
        self.assertTrue(config['evaluation']['all_eligible_retained_test_sections'])
        commands = pipeline.commands(Path('/tmp/isolated'), config)
        text = str(commands)
        for forbidden in ('runs/nyc_all_full', 'runs/nyc_all_dpo_intermediate', 'model_supervised', 'ppo_reward'):
            self.assertNotIn(forbidden, text)
        self.assertIn('/tmp/isolated/pretrain/model_best.pt', text)
        smoke = pipeline.configuration('/tmp/raw', True)
        self.assertEqual(smoke['evaluation'], config['evaluation'])
        self.assertEqual(smoke['preprocessing'], config['preprocessing'])
        self.assertEqual(smoke['pretraining']['batch_size'], 64)
        self.assertIn('Integration', smoke['assessment'])

    def test_explicit_text_root_does_not_read_repository_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'Trajs_NYCAll.txt').write_text('own run')
            config = get_base_config()
            config.data.dataset, config.data.root = 'NYCAll', str(root)
            self.assertEqual(load_trajectory_data(config), 'own run')

    def fixture(self, root):
        raw = root/'raw'
        raw.mkdir()
        for name in ('nyc.geo', 'nyc.rel', 'nyc_mm_train.csv', 'nyc_mm_test.csv'):
            (raw/name).write_text(name)
        args = argparse.Namespace(input_dir=raw, output_dir=root/'run', smoke=True,
                                  resume=False, stop_after='prepare')
        return args

    def runner(self, args, fail=False):
        # A real subprocess writes one exclusive artifact. Only the scientific
        # stage and its output validator are replaced; journal/restart are real.
        artifact = args.output_dir/'payload'
        code = (f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok'); "
                + ('raise SystemExit(3)' if fail else ''))
        command = [sys.executable, '-c', code]
        return contextlib.ExitStack(), artifact, command

    def invoke(self, args, artifact, command):
        with patch.object(pipeline, 'STAGES', ('prepare',)), \
             patch.object(pipeline, 'commands', return_value={'prepare': command}), \
             patch.object(pipeline, 'outputs', return_value=[artifact]), \
             patch.object(pipeline, 'check_stage'), contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_pipeline(args)

    def test_real_stage_restart_skips_and_refuses_overwrite_changed_artifact_input_config_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            _, artifact, command = self.runner(args)
            self.invoke(args, artifact, command)
            journal = (args.output_dir/'stage_status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                self.invoke(args, artifact, command)
            args.resume = True
            self.invoke(args, artifact, command)
            self.assertEqual((args.output_dir/'stage_status.json').read_bytes(), journal)
            artifact.write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.invoke(args, artifact, command)
            artifact.write_text('ok')
            source = args.input_dir/'nyc.geo'
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                self.invoke(args, artifact, command)
            source.write_text('nyc.geo')
            args.smoke = False
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                self.invoke(args, artifact, command)
            args.smoke = True
            with patch.object(pipeline, 'source_paths', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.invoke(args, artifact, command)

    def test_failure_is_recorded_and_partial_restart_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            _, artifact, command = self.runner(args, fail=True)
            with self.assertRaisesRegex(ValueError, 'exited 3'):
                self.invoke(args, artifact, command)
            state = pipeline.read(args.output_dir/'stage_status.json')['prepare']
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['exit_code'], 3)
            args.resume = True
            with self.assertRaisesRegex(ValueError, 'partial/failed'):
                self.invoke(args, artifact, command)

    def test_stage_success_without_valid_outputs_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            artifact = args.output_dir/'missing'
            with self.assertRaisesRegex(ValueError, 'Missing stage artifact'):
                self.invoke(args, artifact, [sys.executable, '-c', 'pass'])


if __name__ == '__main__':
    unittest.main()
