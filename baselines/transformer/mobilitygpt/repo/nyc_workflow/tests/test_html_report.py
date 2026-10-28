"""Offline presentation contracts using saved synthetic JSON, never model execution."""
import contextlib
import hashlib
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote

from nyc_workflow import html_report as page


class Document(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.tags, self.links, self.cells, self.ids, self.tables = [], [], [], set(), []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if 'id' in attributes:
            self.ids.add(attributes['id'])
        if tag == 'a':
            self.links.append(attributes.get('href', ''))
        if tag == 'table':
            self.tables.append(attributes.get('data-kind'))
        if tag == 'td' and 'data-metric' in attributes:
            self.cells.append((self.tables[-1], attributes))

    def handle_endtag(self, tag):
        if tag == 'table':
            self.tables.pop()


class HTMLReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)/'run'; self.run.mkdir()
        self.config = dict(smoke=False, assessment='Full saved assessment',
            dpo=dict(selected_update=100), evaluation=dict(prompt_roads=[1,3], seeds=[42,43,44]))
        self.selected = 'selected_dpo_100'
        self.paired = dict(per_seed={}, verification=dict(actual_samples=24, failed_cases=0))
        rows = []
        for label in ('pretrained', self.selected):
            dpo = label == self.selected
            self.paired['per_seed'][label] = {}
            for prompt in (1,3):
                self.paired['per_seed'][label][str(prompt)] = {}
                for seed in (42,43,44):
                    self.paired['per_seed'][label][str(prompt)][str(seed)] = dict(
                        samples=2, road_count_js_distance=.3408055555555556 if dpo else .38813577296724533,
                        distance_m_js_distance=.3881966666666667 if dpo else .4138016183351666,
                        repeated_road_route_rate=.342 if dpo else .31, immediate_stop_rate=.05,
                        length_cap_rate=0., road_count=dict(mean=13.5), distance_m=dict(mean=3555.5))
                    values = {key:.2 + (.01 if dpo else 0) for key,_,_,_ in page.CANONICAL}
                    values.update(off_network_rate=None, mia_auc=.5, uniqueness=.98, rsc=1.)
                    rows.append(dict(checkpoint=label, prompt_length=prompt, seed=seed, **values,
                        unavailable_reasons=dict(off_network_rate='Canonical network not supplied or file missing')))
        self.canonical = dict(status='incomplete', complete=False, results=rows,
            network=dict(unavailable_reason='Canonical network not supplied'))
        self.selection = dict(selected_update=100, checkpoint=dict(sha256='recorded-only-checksum'))
        self.stages = {name:dict(status='complete') for name in ('preflight','pretrain','dpo','evaluate','canonical_metrics','report')}
        self.save_all()
        (self.run/'training').mkdir(); (self.run/'training/model_step_000100.pt').write_bytes(b'fixture, not a model')
        (self.run/'model_selected.pt').symlink_to('training/model_step_000100.pt')
        (self.run/'pretrain').mkdir(); (self.run/'pretrain/model_best.pt').write_bytes(b'fixture, not a model')
        (self.run/'report.md').write_text('Existing markdown is immutable.\n')

    def save(self, name, value):
        path = self.run/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2)+'\n')

    def save_all(self):
        self.save('pipeline_config.json', self.config)
        self.save('stage_status.json', self.stages)
        self.save('selection.json', self.selection)
        self.save('evaluation/metrics.json', self.paired)
        self.save('canonical_metrics/results.json', self.canonical)
        self.save('report.json', dict(passed=True, pipeline_complete=True, configuration=self.config,
            selection=self.selection, evaluation=self.paired, canonical_metrics=self.canonical))

    def document(self):
        path = page.write_html(self.run)
        text = path.read_text()
        return text, Document(text)

    def test_full_status_selected_checkpoint_and_fixed_reporting_choices(self):
        text, parsed = self.document()
        self.assertIn('Training completed; 11 of 12 metrics available', text)
        self.assertIn('Full training', text)
        self.assertIn('selected update 100', text)
        self.assertIn('training/model_step_000100.pt', text)
        self.assertIn('One-road prompt, seed 42', text)
        self.assertIn('not the best-performing seed', text)
        self.assertIn('model_selected.pt', parsed.links)
        self.assertIn('pretrain/model_best.pt', parsed.links)
        self.assertTrue({'status','comparison','meaning','developer'} <= parsed.ids)
        self.assertIn('revisiting a road increases', text)
        self.assertIn('not a privacy guarantee', text)
        self.assertIn('No statistical significance or overall winning model', text)

    def test_smoke_label_and_selected_update_are_not_historical_full_results(self):
        self.config['smoke'] = True
        self.config['dpo']['selected_update'] = 1
        self.selection['selected_update'] = 1
        self.paired['per_seed']['selected_dpo_1'] = self.paired['per_seed'].pop(self.selected)
        for row in self.canonical['results']:
            if row['checkpoint'] == self.selected:
                row['checkpoint'] = 'selected_dpo_1'
        (self.run/'training/model_step_000001.pt').write_bytes(b'smoke fixture, not a model')
        (self.run/'model_selected.pt').unlink()
        (self.run/'model_selected.pt').symlink_to('training/model_step_000001.pt')
        self.save_all()
        text, parsed = self.document()
        self.assertIn('Smoke run — integration diagnostics only', text)
        self.assertIn('selected update 1.', text)
        self.assertIn('not full-training quality evidence', text)
        self.assertNotIn('<dd>Full training</dd>', text)
        selected = [attrs for _,attrs in parsed.cells if attrs['data-checkpoint'] == 'selected_dpo_1']
        self.assertTrue(any(attrs['data-value'] != 'null' for attrs in selected))

    def test_all_displayed_metric_values_exactly_match_saved_json(self):
        _, parsed = self.document()
        canonical = {(row['checkpoint'],row['prompt_length'],row['seed']):row for row in self.canonical['results']}
        self.assertEqual(len(parsed.cells), (len(page.PAIRED)+len(page.CANONICAL))*2*7)
        for kind, attributes in parsed.cells:
            label, prompt, seed = attributes['data-checkpoint'], int(attributes['data-prompt']), int(attributes['data-seed'])
            source = (self.paired['per_seed'][label][str(prompt)][str(seed)] if kind == 'paired'
                      else canonical[label,prompt,seed])
            expected = page.value_at(source, attributes['data-metric'])
            self.assertEqual(json.loads(attributes['data-value']), expected)
            if expected is not None:
                self.assertEqual(attributes['title'], 'Saved value: '+json.dumps(expected))

    def test_missing_network_does_not_mean_training_failed_or_zero_score(self):
        text, parsed = self.document()
        self.assertNotIn('<p class="status">incomplete', text)
        self.assertIn('Off-network reason: Canonical network not supplied', text)
        self.assertIn('Prepared geometry is not a substitute', text)
        off = [attrs for _,attrs in parsed.cells if attrs['data-metric'] == 'off_network_rate']
        self.assertTrue(off and all(attrs['data-value'] == 'null' for attrs in off))

    def test_complete_network_results_are_labeled_twelve_of_twelve(self):
        self.canonical.update(status='complete', complete=True)
        for row in self.canonical['results']:
            row['off_network_rate'] = .01234
            row['unavailable_reasons'] = {}
        self.save_all()
        text, _ = self.document()
        self.assertIn('Training completed; 12 of 12 metrics available', text)
        self.assertNotIn('Off-network reason:', text)

    def test_missing_or_nonfinite_saved_values_are_explicitly_unavailable(self):
        for row in self.canonical['results']:
            row['mia_auc'] = None
            row['normalized_dtw_km'] = float('nan')
            row['unavailable_reasons']['mia_auc'] = 'No eligible candidates'
        self.save_all()
        text, parsed = self.document()
        self.assertIn('Training completed; 9 of 12 metrics available', text)
        self.assertIn('No eligible candidates', text)
        missing = [attrs for _,attrs in parsed.cells if attrs['data-metric'] in ('mia_auc','normalized_dtw_km')]
        self.assertTrue(all(attrs['data-value'] == 'null' for attrs in missing))

    def test_html_escaping_of_all_saved_text_and_safe_file_links(self):
        payload = '</pre><script src="https://invalid.example/evil.js">alert(1)</script><img src=x onerror=alert(1)>'
        self.config['untrusted'] = payload
        self.selection['checkpoint']['sha256'] = payload
        self.stages[payload] = dict(status='complete')
        for row in self.canonical['results']:
            row['unavailable_reasons']['off_network_rate'] = payload
        self.save_all()
        text, parsed = self.document()
        self.assertIn('&lt;script', text)
        self.assertFalse(any(tag in ('script','img','iframe','svg') for tag,_ in parsed.tags))
        self.assertFalse(any(key.lower().startswith('on') for _,attrs in parsed.tags for key in attrs))
        unsafe = self.run/'javascript:alert(1).json'; unsafe.write_text('{}')
        link = page.file_link(self.run, unsafe.name)
        self.assertIn('javascript%3A', link)
        self.assertNotIn('href="javascript:', link)

    def test_page_is_self_contained_and_every_link_exists(self):
        text, parsed = self.document()
        self.assertFalse(any(tag in ('script','link','img','iframe','object','embed','audio','video','base') for tag,_ in parsed.tags))
        self.assertNotIn('url(', page.CSS.lower())
        self.assertNotIn('@import', page.CSS.lower())
        self.assertIn("connect-src 'none'", text)
        for href in parsed.links:
            if href.startswith('#'):
                self.assertIn(href[1:], parsed.ids)
            else:
                self.assertFalse(href.startswith(('/', 'http:', 'https:', '//', 'data:', 'javascript:')))
                self.assertTrue((self.run/unquote(href)).exists(), href)
        self.assertIn('not available locally', text)  # optional logs/source/CSV absent in fixture

    def test_absent_optional_results_use_saved_fallback_or_unavailable(self):
        (self.run/'canonical_metrics/results.json').unlink()
        (self.run/'stage_status.json').unlink()
        text, _ = self.document()
        self.assertIn('using its saved canonical_metrics record in report.json', text)
        self.assertIn('Stage-by-stage journal unavailable', text)
        self.assertIn('11 of 12 metrics available', text)

    def test_absent_canonical_results_are_not_invented(self):
        (self.run/'canonical_metrics/results.json').unlink()
        report = json.loads((self.run/'report.json').read_text()); report.pop('canonical_metrics')
        self.save('report.json', report)
        text, parsed = self.document()
        self.assertIn('Training completed; canonical metric results unavailable', text)
        raw = [attrs for kind,attrs in parsed.cells if kind == 'canonical']
        self.assertTrue(all(attrs['data-value'] == 'null' for attrs in raw))

    def test_no_overwrites_and_only_one_new_file(self):
        def hashes():
            return {str(p.relative_to(self.run)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in self.run.rglob('*') if p.is_file()}
        before = hashes()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(page.main(['--run-dir',str(self.run)]), 0)
        self.assertEqual(output.getvalue().strip(), str(self.run/'report.html'))
        after = hashes()
        self.assertEqual(set(after)-set(before), {'report.html'})
        self.assertTrue(all(after[name] == value for name,value in before.items()))
        with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite'):
            page.write_html(self.run)
        self.assertEqual(hashes(), after)

    def test_refuses_existing_report_symlink_and_incomplete_run(self):
        (self.run/'report.html').symlink_to('does-not-exist')
        with self.assertRaisesRegex(FileExistsError, 'overwrite'):
            page.write_html(self.run)
        (self.run/'report.html').unlink()
        self.stages['dpo']['status'] = 'failed'; self.save_all()
        with self.assertRaisesRegex(ValueError, 'Run is not completed'):
            page.write_html(self.run)
        self.assertFalse((self.run/'report.html').exists())

    def test_final_report_stage_can_render_itself_without_fabricating_journal_status(self):
        self.stages['report']['status'] = 'running'; self.save_all()
        with self.assertRaisesRegex(ValueError, 'not completed'):
            page.write_html(self.run)
        page.write_html(self.run, finalizing=True)
        text = (self.run/'report.html').read_text()
        self.assertIn('Training completed; 11 of 12 metrics available', text)
        self.assertIn('journal records final artifact sealing', text)
        self.assertEqual(json.loads((self.run/'stage_status.json').read_text())['report']['status'], 'running')

    def test_conflicting_selection_or_duplicate_group_fails_without_writing(self):
        self.selection['selected_update'] = 250; self.save_all()
        with self.assertRaisesRegex(ValueError, 'selection.*disagree'):
            page.write_html(self.run)
        self.selection['selected_update'] = 100
        self.canonical['results'].append(dict(self.canonical['results'][0])); self.save_all()
        with self.assertRaisesRegex(ValueError, 'Duplicate canonical metric group'):
            page.write_html(self.run)
        self.assertFalse((self.run/'report.html').exists())

    def test_checkpoint_link_absent_or_outside_run_is_not_emitted(self):
        (self.run/'model_selected.pt').unlink()
        outside = Path(self.temporary.name)/'outside.pt'; outside.write_bytes(b'fixture')
        (self.run/'model_selected.pt').symlink_to(outside)
        text, parsed = self.document()
        self.assertNotIn('model_selected.pt', parsed.links)
        self.assertNotIn(str(outside), parsed.links)
        self.assertIn('not available locally', text)

    def test_distribution_formulas_and_file_guide_are_distinguished(self):
        text, _ = self.document()
        for phrase in ('square root of base-2', 'mixes raw counts first', 'not standard equal-weight JSD',
                       'Reports and CSVs', 'Checkpoints', 'Prepared data and preferences',
                       'Logs, snapshots and manifests', 'Most users need only', 'no installation, internet or server'):
            self.assertIn(phrase, text)

    def test_cli_imports_no_scientific_dependencies_and_prints_absolute_path(self):
        code = ('import sys; from nyc_workflow.__main__ import main; '
                f'assert main(["summarize","--run-dir",{str(self.run)!r}]) == 0; '
                'assert not any(n.split(".")[0] in {"torch","numpy","scipy","geopandas","fastdtw"} for n in sys.modules)')
        result = subprocess.run([sys.executable, '-I', '-c', code], cwd=self.temporary.name,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.run/'report.html'))

    def test_pipeline_seals_and_checks_html_as_a_report_artifact(self):
        from nyc_workflow import pipeline
        self.assertIn(self.run/'report.html', pipeline.outputs(self.run, 'report'))
        with self.assertRaisesRegex(ValueError, 'HTML report missing'):
            pipeline.check_stage(self.run, 'report', self.config)
        page.write_html(self.run)
        pipeline.check_stage(self.run, 'report', self.config)


if __name__ == '__main__':
    unittest.main()
