"""Unattended orchestration and internal-output contracts, without NYC training."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nyc_workflow import pipeline, preflight, runtime


class UnattendedWorkflowTests(unittest.TestCase):
    def fixture(self, root, stop_after='report'):
        raw = root/'raw'; raw.mkdir()
        for name in ('nyc.geo', 'nyc.rel', 'nyc_mm_train.csv', 'nyc_mm_test.csv'):
            (raw/name).write_text(name)
        return argparse.Namespace(input_dir=raw, output_dir=root/'run', smoke=True,
            canonical_network=None, network_layer='roads', network_mode_column=None,
            metric_jobs=1, resume=False, stop_after=stop_after)

    def invoke(self, args, run, fail=None):
        order = run/'observed_order.txt'
        commands = {}
        for stage in pipeline.STAGES:
            code = (f"from pathlib import Path; p=Path({str(order)!r}); "
                    f"p.open('a').write({stage!r}+'\\n'); "
                    f"Path({str(run/(stage+'.payload'))!r}).write_text({stage!r}); "
                    + ('raise SystemExit(7)' if stage == fail else ''))
            commands[stage] = [sys.executable, '-c', code]
        with patch.object(pipeline, 'commands', return_value=commands), \
             patch.object(pipeline, 'outputs', side_effect=lambda directory, stage: [directory/(stage+'.payload')]), \
             patch.object(pipeline, 'check_stage'), contextlib.redirect_stdout(io.StringIO()):
            pipeline.run_pipeline(args)
        return order

    def test_default_timestamped_output_and_collision_refusal(self):
        first, second = pipeline.default_output(), pipeline.default_output()
        self.assertEqual(first.parent, runtime.PACKAGE/'output')
        self.assertRegex(first.name, r'^\d{8}T\d{6}\.\d{6}Z-[a-f0-9]{6}$')
        self.assertNotEqual(first, second)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); args = self.fixture(root, 'preflight')
            run = root/'nyc_workflow/output/fixed'
            args.output_dir = None
            with patch.object(pipeline, 'default_output', return_value=run):
                self.invoke(args, run)
                with self.assertRaises(FileExistsError):
                    self.invoke(args, run)
            self.assertEqual(pipeline.read(run/'stage_status.json')['preflight']['status'], 'complete')

    def test_input_overlap_and_source_aliases_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); raw = root/'raw'; raw.mkdir()
            alias = root/'raw_alias'; alias.symlink_to(raw, target_is_directory=True)
            for output in (raw, raw/'run', root, alias/'run'):
                with self.assertRaisesRegex(ValueError, 'separate'):
                    runtime.require_output(output, [raw])
            source_alias = root/'source_alias'; source_alias.symlink_to(runtime.PACKAGE, target_is_directory=True)
            for output in (runtime.PACKAGE, source_alias, runtime.PACKAGE.parent):
                with self.assertRaisesRegex(ValueError, 'source files'):
                    runtime.require_output(output)
            self.assertEqual(runtime.require_output(runtime.PACKAGE/'custom_run'), runtime.PACKAGE/'custom_run')

    def test_resume_needs_an_explicit_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary)); args.resume = True; args.output_dir = None
            with self.assertRaisesRegex(ValueError, 'original explicit --output-dir'):
                pipeline.run_pipeline(args)

    def test_generated_python_and_snapshots_do_not_change_source_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); modules = {}
            for name in ('nyc_workflow', 'mobilitygpt', 'finetuners', 'datasets', 'preprocessing'):
                directory = root/name; directory.mkdir()
                (directory/'__init__.py').write_text('# source\n')
                modules[name] = SimpleNamespace(__file__=str(directory/'__init__.py'))
            package = root/'nyc_workflow'
            for name in ('requirements.txt', 'notebook_provenance.json'):
                (package/name).write_text('source provenance')
            (root/'pyproject.toml').write_text('[project]')
            with patch.object(runtime, 'PACKAGE', package), \
                 patch.object(runtime.importlib, 'import_module', side_effect=modules.__getitem__):
                before = pipeline.inventory(runtime.source_entries().values())
                for folder in ('output/run/source/nyc_workflow', '.venv/lib', '__pycache__', 'build', 'custom_run'):
                    directory = package/folder; directory.mkdir(parents=True, exist_ok=True)
                    (directory/'__init__.py').write_text('# generated, not source\n')
                    (directory/'generated.py').write_text('raise AssertionError("must not import")\n')
                (package/'custom_run/.pipeline.lock').touch()
                snapshot = package/'output/another_run/source'
                runtime.snapshot_sources(snapshot)
                after = pipeline.inventory(runtime.source_entries().values())
                self.assertEqual(before, after)
                self.assertTrue((snapshot/'nyc_workflow/__init__.py').is_file())
                self.assertFalse(any('output' in Path(name).parts for name in runtime.source_entries()))
                self.assertEqual(len(list(snapshot.rglob('*.py'))), 5)

    def test_all_stages_chain_and_completed_resume_skips_every_stage(self):
        expected = ('preflight', 'prepare', 'gravity', 'verify', 'safety', 'pretrain',
                    'preferences', 'dpo', 'select', 'evaluation_audit', 'evaluate', 'canonical_metrics', 'report')
        self.assertEqual(pipeline.STAGES, expected)
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            order = self.invoke(args, args.output_dir)
            self.assertEqual(order.read_text().splitlines(), list(expected))
            before = (args.output_dir/'stage_status.json').read_bytes()
            args.resume = True
            self.invoke(args, args.output_dir)
            self.assertEqual(order.read_text().splitlines(), list(expected))
            self.assertEqual((args.output_dir/'stage_status.json').read_bytes(), before)
            (args.output_dir/'canonical_metrics.payload').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.invoke(args, args.output_dir)

    def test_failure_stops_downstream_stages_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'safety exited 7'):
                self.invoke(args, args.output_dir, fail='safety')
            stages = (args.output_dir/'observed_order.txt').read_text().splitlines()
            self.assertEqual(stages, list(pipeline.STAGES[:5]))
            self.assertFalse((args.output_dir/'pretrain.payload').exists())
            self.assertEqual(pipeline.read(args.output_dir/'stage_status.json')['safety']['status'], 'failed')
            args.resume = True
            with self.assertRaisesRegex(ValueError, 'partial/failed'):
                self.invoke(args, args.output_dir)

    def test_commands_are_run_local_and_automatically_allow_missing_network(self):
        config = pipeline.configuration('/tmp/explicit_raw')
        commands = pipeline.commands(Path('/tmp/fresh_run'), config)
        self.assertEqual(set(commands), set(pipeline.STAGES))
        metrics = commands['canonical_metrics']
        self.assertIn('--allow-missing-network', metrics)
        self.assertIn('/tmp/fresh_run/canonical_metrics', metrics)
        self.assertIn('/tmp/fresh_run/pretrain/model_best.pt', commands['dpo'])
        for forbidden in ('historical_bundle', 'handoff/', 'archives/', 'evaluation_outputs/', '202610', 'diagnose_', 'compare_'):
            self.assertNotIn(forbidden, str(commands))
        config = pipeline.configuration('/tmp/raw', canonical_network='/tmp/canonical.gpkg')
        supplied = pipeline.commands(Path('/tmp/new'), config)['canonical_metrics']
        self.assertIn('--canonical-network', supplied)
        self.assertNotIn('--allow-missing-network', supplied)

    def test_missing_network_preflight_is_incomplete_without_skipping_training(self):
        config = pipeline.configuration('/tmp/raw', True)
        with patch.object(preflight, 'validate_raw', return_value={'parent_overlap': 0}), \
             patch.object(preflight, 'validate_cuda', return_value={'matrix_multiply': 'passed'}):
            result = preflight.validate(config)
        self.assertTrue(result['passed'])
        self.assertFalse(result['canonical_network']['complete'])
        self.assertIn('11 network-independent metrics', result['canonical_network']['unavailable_reason'])
        self.assertIn('pretrain', pipeline.commands(Path('/tmp/new'), config))

    def test_invalid_supplied_network_fails_before_raw_processing_or_cuda(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        with tempfile.TemporaryDirectory() as temporary:
            network = Path(temporary)/'bad.gpkg'; network.write_bytes(b'fixture')
            config = pipeline.configuration('/tmp/raw', canonical_network=network)
            roads = gpd.GeoDataFrame(geometry=[LineString([(-74, 40.7), (-73.9, 40.8)])])
            with patch('geopandas.read_file', return_value=roads), \
                 patch.object(preflight, 'validate_raw') as raw, patch.object(preflight, 'validate_cuda') as cuda:
                with self.assertRaisesRegex(ValueError, 'no CRS'):
                    preflight.validate(config)
                raw.assert_not_called(); cuda.assert_not_called()

    def test_canonical_completion_contract_and_no_fabricated_off_network_score(self):
        from nyc_workflow.metrics import METRICS
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary); (run/'canonical_metrics').mkdir()
            config = pipeline.configuration(run/'raw')
            row = {name: None if name == 'off_network_rate' else .1 for name in METRICS}
            row['unavailable_reasons'] = {'off_network_rate': 'Canonical network unavailable'}
            result = dict(status='incomplete', complete=False, input_hashes_preserved=True,
                          results=[dict(row) for _ in range(12)])
            path = run/'canonical_metrics/results.json'; path.write_text(json.dumps(result))
            pipeline.check_stage(run, 'canonical_metrics', config)
            result['results'][0]['off_network_rate'] = 0
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Fabricated off-network'):
                pipeline.check_stage(run, 'canonical_metrics', config)


if __name__ == '__main__':
    unittest.main()
