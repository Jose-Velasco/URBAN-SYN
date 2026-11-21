"""Installed-module handoffs without NYC training or trajectory generation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nyc_workflow import pipeline, preprocessing, pretraining, dpo, evaluation
from nyc_workflow.runtime import require_output, PACKAGE


class PackageHandoffs(unittest.TestCase):
    def test_every_stage_parser_receives_explicit_run_local_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, run = root/'raw', root/'run'
            raw.mkdir(); run.mkdir()
            config = pipeline.configuration(raw)
            (run/'pipeline_config.json').write_text(json.dumps(config))
            commands = pipeline.commands(run, config)
            with patch.object(preprocessing, 'prepare') as prepare:
                self.assertEqual(preprocessing.main(commands['prepare'][4:]), 0)
                self.assertEqual(prepare.call_args.args[:2], (raw, run/'prepared'))
            for stage in ('preflight', 'gravity', 'verify', 'select', 'report'):
                with patch.object(pipeline, 'worker') as worker:
                    pipeline.main(commands[stage][4:])
                    self.assertEqual(worker.call_args.args, (stage, run))
            for stage in ('safety', 'pretrain'):
                with patch.object(pretraining, 'run') as train:
                    pretraining.main(commands[stage][4:])
                    args = train.call_args.args[0]
                    self.assertEqual(args.data_root, run/'prepared')
                    self.assertEqual(args.output, run/stage)
            for stage in ('preferences', 'dpo'):
                with patch.object(dpo, 'child', return_value=0) as child:
                    self.assertEqual(dpo.main(commands[stage][4:]), 0)
                    self.assertEqual(child.call_args.args[-2:],
                                     (run/'pretrain/model_best.pt', run/'prepared'))
            with patch.object(evaluation, 'audit') as audit:
                evaluation.main(commands['evaluation_audit'][4:])
                self.assertEqual(audit.call_args.args, (run/'evaluation',))
                self.assertEqual(evaluation.CHECKPOINTS['pretrained'], run/'pretrain/model_best.pt')
                self.assertEqual(evaluation.RAW, raw)

    def test_selection_and_report_handoff_is_local_to_the_supplied_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); run = root/'run'; run.mkdir()
            config = pipeline.configuration(root/'raw')
            (run/'pipeline_config.json').write_text(json.dumps(config))
            (run/'training').mkdir()
            target = run/'training/model_step_000100.pt'
            target.write_bytes(b'fixture checkpoint identity; not a model')
            pipeline.worker('select', run)
            self.assertEqual((run/'model_selected.pt').resolve(), target)
            self.assertEqual(json.loads((run/'selection.json').read_text())['selected_update'], 100)
            for name in ('preflight.json', 'verification.json', 'pretrain/results.json', 'preferences/dataset_audit.json',
                         'training/result.json', 'evaluation/metrics.json', 'stage_status.json'):
                path = run/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'fixture': True}))
            row = dict(road_count_js_distance=.2, distance_m_js_distance=.3, repeated_road_route_rate=.4)
            (run/'evaluation/metrics.json').write_text(json.dumps({'per_seed': {'pretrained': {'1': {'42': row}}}}))
            (run/'canonical_metrics').mkdir()
            from nyc_workflow.metrics import METRICS
            canonical = dict(complete=False, status='incomplete', results=[
                dict(checkpoint='pretrained', prompt_length=1, seed=42,
                     **{name: None if name == 'off_network_rate' else .1 for name in METRICS})])
            (run/'canonical_metrics/results.json').write_text(json.dumps(canonical))
            (run/'stage_status.json').write_text(json.dumps({
                'pretrain': {'status': 'complete'}, 'dpo': {'status': 'complete'},
                'canonical_metrics': {'status': 'complete'}, 'report': {'status': 'running'}}))
            pipeline.worker('report', run)
            report = json.loads((run/'report.json').read_text())
            self.assertEqual(report['artifacts']['selected_dpo'], str(run/'model_selected.pt'))
            self.assertEqual(report['configuration'], config)
            self.assertIn('per_seed', json.loads((run/'metrics.json').read_text()))
            self.assertFalse(report['canonical_assessment']['complete'])
            text = (run/'report.md').read_text()
            self.assertIn('(evaluation/report.md)', text)
            self.assertIn('(canonical_metrics/report.md)', text)
            self.assertIn('model_selected.pt', text)
            self.assertTrue((run/'report.html').is_file())

    def test_output_guard_follows_symlinks_and_protects_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alias = root/'alias'; alias.symlink_to(PACKAGE, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'source files'):
                require_output(alias)
            self.assertEqual(require_output(alias/'new_run'), PACKAGE/'new_run')
            self.assertEqual(require_output(root/'results'), root/'results')

    def test_run_output_is_optional_but_standalone_metric_output_is_required(self):
        from nyc_workflow import metrics
        with patch.object(pipeline, 'run_pipeline') as run:
            pipeline.main(['--input-dir', '/tmp/raw'])
            self.assertIsNone(run.call_args.args[0].output_dir)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            metrics.main(['--run-dir', '/tmp/run', '--raw-data-dir', '/tmp/raw'])
        self.assertEqual(result.exception.code, 2)
        with self.assertRaisesRegex(ValueError, 'Explicit pretrained'):
            dpo.config_for(Path('/tmp/dpo'))
