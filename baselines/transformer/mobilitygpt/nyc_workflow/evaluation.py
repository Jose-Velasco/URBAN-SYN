"""Locked, evaluation-only comparison of pretrained and selected update-100 NYCAll.

Run ``... audit`` before ``... evaluate`` (same fresh --output). Evaluation cannot
resume a partial generation file or overwrite results. No validation cache is used.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from nyc_workflow.paired import (PROMPTS, SEEDS, describe, fingerprint,
    generate_route, require, route_distance, route_quality, select_validation,
    stable_seed, summarize, verify_pairing, write_json)
from datasets.char_dataset import CharDataset
from datasets.data_utils import load_geo_data, od_pair_to_adjacency_matrix
from nyc_workflow.data import GravityTrajectoryDataset, DeterministicTrajectoryWindows
from mobilitygpt.config_utils import CfgNode
from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from nyc_workflow.preprocessing import verify_export

from .runtime import require_output, source_entries, module_command, git_record

DATA = RUN = PRETRAINING = TEXT = PREFERENCES = RAW = None
LABELS = ('pretrained', 'selected_dpo_100')
CHECKPOINTS = {}
SELECTED_UPDATE = 100
PREFERENCE_COUNT = 1000
ASSESSMENT = 'Explicit saved-run assessment'


def configure_pipeline(run, raw_data_dir):
    """Resolve only this run's artifacts; no historical fallbacks."""
    global DATA, RUN, LABELS, CHECKPOINTS, PRETRAINING, TEXT, PREFERENCES
    global SELECTED_UPDATE, PREFERENCE_COUNT, ASSESSMENT
    global RAW
    RUN = Path(run).resolve()
    config = json.loads((RUN / 'pipeline_config.json').read_text())
    RAW = Path(raw_data_dir).resolve()
    DATA = RUN / 'prepared/NYCAll-Taxi'
    TEXT = RUN / 'prepared/Trajs_NYCAll.txt'
    PRETRAINING = RUN / 'pretrain'
    PREFERENCES = RUN / 'preferences'
    SELECTED_UPDATE = config['dpo']['selected_update']
    PREFERENCE_COUNT = config['dpo']['preferences']
    ASSESSMENT = config['assessment']
    LABELS = ('pretrained', f'selected_dpo_{SELECTED_UPDATE}')
    CHECKPOINTS = {'pretrained': PRETRAINING / 'model_best.pt',
                   LABELS[1]: RUN / 'model_selected.pt'}


def read_rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and len(set(reader.fieldnames)) == len(reader.fieldnames),
                'Missing or duplicate CSV columns')
        rows = list(reader)
    require(rows and all(None not in row and all(v is not None for v in row.values()) for row in rows),
            'Empty or malformed trajectory CSV')
    return rows


def load_sections(csv_path, provenance_path, mapping, edges, split):
    """Require one-to-one CSV/provenance identity, roads, times and observations."""
    rows = read_rows(csv_path)
    by_id = {r['traj_id']: r for r in rows}
    require(len(by_id) == len(rows), 'Duplicate trajectory IDs')
    found, parents, statuses = set(), set(), Counter()
    with Path(provenance_path).open() as stream:
        for line in stream:
            trip = json.loads(line)
            parent = trip['parent_traj_id']
            require(trip['split'] == split and parent not in parents, 'Wrong split or duplicate parent')
            parents.add(parent)
            for section in trip['sections']:
                statuses[section['status']] += 1
                if section['status'] != 'retained':
                    require(section['export_traj_id'] is None, 'Discarded section exported')
                    continue
                identity = section['export_traj_id']
                require(identity in by_id and identity not in found, 'CSV/provenance identity mismatch')
                found.add(identity)
                row = by_id[identity]
                start, stop = section['node_range']
                nodes = trip['nodes'][start:stop]
                require(row['parent_traj_id'] == parent and int(row['section_index']) == section['section_index'],
                        'CSV/provenance parent or section mismatch')
                require(row['original_rid_list'] == ','.join(str(n['road_id']) for n in nodes),
                        'CSV/provenance original roads mismatch')
                require(row['rid_list'] == ','.join(str(mapping[n['road_id']]) for n in nodes),
                        'CSV/provenance mapped roads mismatch')
                require(row['time_list'] == ','.join(n['timestamp'] for n in nodes) and
                        json.loads(row['observation_ranges']) == [n['observation_range'] for n in nodes],
                        'CSV/provenance observations mismatch')
    require(found == set(by_id), 'Missing provenance')
    valid_ids = set(mapping.values())
    for row in rows:
        roads = [int(t) for t in row['rid_list'].split(',')]
        require(len(roads) >= 2 and set(roads) <= valid_ids, 'Invalid road IDs or singleton')
        require(all(pair in edges for pair in zip(roads, roads[1:])), 'Disconnected reference')
    return rows, parents, dict(statuses)


def eligibility(rows):
    """Manifest all sections, including every no-recorded-continuation exclusion."""
    return [{'trajectory_index': i, 'trajectory_id': r['traj_id'], 'parent_id': r['parent_traj_id'],
             'reference_roads': list(map(int, r['rid_list'].split(','))),
             'eligible_prompt_lengths': [p for p in PROMPTS if len(r['rid_list'].split(',')) > p],
             'exclusions': {str(p): 'no_recorded_continuation' for p in PROMPTS
                            if len(r['rid_list'].split(',')) <= p}}
            for i, r in enumerate(rows)]


def sequence_overlap(train, test):
    groups = defaultdict(list)
    for row in train:
        groups[tuple(map(int, row['rid_list'].split(',')))].append(row)
    matches = []
    for row in test:
        sources = groups.get(tuple(map(int, row['rid_list'].split(','))), [])
        if sources:
            matches.append({'test_trajectory_id': row['traj_id'], 'test_parent_id': row['parent_traj_id'],
                            'training_trajectory_ids': [r['traj_id'] for r in sources],
                            'training_parent_ids': [r['parent_traj_id'] for r in sources]})
    return {'test_sections_matching_training_sequences': len(matches),
            'test_parents_with_matching_sequences': len({r['test_parent_id'] for r in matches}),
            'matches': matches, 'duplicates_removed': 0}


def coverage_check(adapter, windows):
    """Check exact target positions, not just a potentially misleading token sum."""
    observed = defaultdict(list)
    for i, (trajectory, offset) in enumerate(windows.windows):
        x, y = windows[i]
        source = adapter.trajectories[trajectory]
        count = min(adapter.block_size, len(source)-1-offset)
        require(torch.equal(x[:count], source[offset:offset+count]) and
                torch.equal(y[:count], source[offset+1:offset+count+1]), 'Incorrect window content')
        require(torch.all(y[count:] == -1).item(), 'Padding target not ignored')
        observed[trajectory].extend(range(offset+1, offset+count+1))
    require(set(observed) == set(range(len(adapter))), 'Missing target trajectory')
    for i, positions in observed.items():
        require(positions == list(range(1, len(adapter.trajectories[i]))), 'Targets not covered once')
    return {'passed': True, 'targets': sum(map(len, observed.values())),
            'windows': len(windows), 'trajectories': len(adapter),
            'includes_first_road_and_terminal_eos': True, 'padding_targets_excluded': True,
            'each_target_position_covered_exactly_once': True}


def paths_to_preserve():
    paths = [*CHECKPOINTS.values(), PRETRAINING/'config.json',
             PRETRAINING/'split.json', RUN/'selection.json',
             RUN/'training/result.json', PREFERENCES/'dataset_audit.json',
             PREFERENCES/'preference_dataset_dpo', TEXT,
             *source_entries().values()]
    paths += sorted(p for p in DATA.rglob('*') if p.is_file())
    metadata = json.loads((DATA/'processing_metadata.json').read_text())
    paths += [RAW/Path(info['path']).name for info in metadata['source_files'].values()]
    return list(dict.fromkeys(paths))


def assert_inputs(protocol):
    for path, before in protocol['input_fingerprints'].items():
        require(fingerprint(path) == before, f'Input changed: {path}')
    require(str(CHECKPOINTS[LABELS[1]].resolve()) == protocol['resolved_selected_checkpoint'],
            'Selected symlink changed')


def recorded_content_matches(path, recorded):
    """Validate saved content identities after relocating a source/artifact handoff.

    Historical path strings stay immutable; new audit locks use current paths.
    """
    current = fingerprint(path)
    return all(current[k] == recorded[k] for k in ('sha256', 'bytes'))


def run_tests(output):
    command = module_command('nyc_workflow', 'check')
    started = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True)
    with (output/'main_tests.log').open('x') as stream:
        stream.write(process.stdout+process.stderr)
    results = [dict(suite='nyc_workflow',command=command,exit_code=process.returncode,
                    elapsed_seconds=time.monotonic()-started)]
    write_json(output/'test_results.json', results)
    require(process.returncode == 0, 'Regression tests failed; see main_tests.log')


def evaluation_output(output):
    output = require_output(output, [RAW])
    # The pipeline owns this fresh subdirectory; standalone assessments use a
    # separate output so they cannot modify any existing saved-run inputs.
    if output != RUN/'evaluation':
        require_output(output, [RUN])
    return output


def audit(output):
    output = evaluation_output(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    selected = CHECKPOINTS[LABELS[1]]
    require(selected.is_symlink() and selected.resolve() == (RUN/f'training/model_step_{SELECTED_UPDATE:06d}.pt').resolve(),
            'Selected checkpoint must match configured update')
    before = {str(p): fingerprint(p) for p in paths_to_preserve()}
    selection = json.loads((RUN/'selection.json').read_text())
    result = json.loads((RUN/'training/result.json').read_text())
    require(selection['selected_update'] == SELECTED_UPDATE and
            recorded_content_matches(selected, selection['checkpoint']) and
            fingerprint(selected)['sha256'] == result['milestone_checkpoints'][str(SELECTED_UPDATE)]['sha256'],
            'Selection/checkpoint hash mismatch')
    require(recorded_content_matches(CHECKPOINTS['pretrained'], result['initial_checkpoint']), 'Pretrained hash mismatch')
    require(recorded_content_matches(PREFERENCES/'preference_dataset_dpo', result['preference_fingerprint']),
            'Preference artifact hash mismatch')
    saved = json.loads((PRETRAINING/'config.json').read_text())
    split = json.loads((PRETRAINING/'split.json').read_text())
    require(saved['data']['dataset'] == split['dataset'] == 'NYCAll' and saved['model']['use_adjacency']
            and saved['data']['block_size'] == saved['model']['block_size'] == 81, 'Incompatible saved config')
    export = verify_export(DATA, TEXT, 'NYCAll')
    geo, vocabulary, _, relations = load_geo_data('NYCAll', DATA.parent)
    require(vocabulary == list(map(str, range(len(vocabulary)))) and
            saved['model']['vocab_size'] == len(vocabulary)+1, 'Vocabulary incompatible')
    edges = {tuple(map(int, pair)) for pair in relations}
    mapping = {int(r['original_id']): int(r['new_id']) for r in read_rows(DATA/'id_mapping.csv')}
    train, train_parents, train_status = load_sections(DATA/'NYCAll_Taxi_trajectory_train.csv',
                    DATA/'train_provenance.jsonl', mapping, edges, 'train')
    test, test_parents, test_status = load_sections(DATA/'NYCAll_Taxi_trajectory_test.csv',
                    DATA/'test_provenance.jsonl', mapping, edges, 'test')
    require(not train_parents & test_parents, 'Parent leakage train/test')
    # Validate the ENTIRE saved internal split, not a newly sampled validation subset.
    select_validation(train, split, count=len(split['validation_indices']))
    require(fingerprint(DATA/'NYCAll_Taxi_trajectory_train.csv')['sha256'] == split['source_sha256'],
            'Training CSV changed since saved split')
    base_train = CharDataset(CfgNode(**saved['data']), TEXT.read_text(), vocabulary)
    lookup = defaultdict(list)
    for i, trajectory in enumerate(base_train.trajs):
        lookup[tuple(base_train.stoi[t] for t in trajectory)].append(i)
    preferences = torch.load(PREFERENCES/'preference_dataset_dpo', map_location='cpu', weights_only=True)
    audited = json.loads((PREFERENCES/'dataset_audit.json').read_text())['records']
    require(len(preferences) == len(audited) == PREFERENCE_COUNT, 'Unexpected preference audit size')
    preference_parents, preference_rows = set(), set()
    for example, record in zip(preferences, audited):
        matches = lookup[tuple(example['prompt_chosen_tokens'].tolist()[4:])]
        require(matches and matches == record['possible_source_rows'], 'Preference audit source mismatch')
        parents = [train[i]['parent_traj_id'] for i in matches]
        require(parents == record['possible_parent_ids'], 'Preference audit parent mismatch')
        preference_parents.update(parents)
        preference_rows.update(matches)
    require(preference_parents <= train_parents and not preference_parents & test_parents,
            'Preference source parent leakage')
    metadata = json.loads((DATA/'processing_metadata.json').read_text())
    for info in metadata['source_files'].values():
        require(recorded_content_matches(RAW/Path(info['path']).name, info), 'Original preprocessing source fingerprint changed')
    counts = metadata['splits']['test']
    require(len(test) == counts['retained_sections'] and
            len({r['parent_traj_id'] for r in test}) == counts['retained_original_trips'] and
            len(test_parents) == counts['original_trips'] and
            sum(len(r['rid_list'].split(',')) for r in test) == counts['retained_road_tokens'],
            'Preprocessing coverage discrepancy')
    manifest = eligibility(test)
    require(route_distance([0,1,0], {0:10.,1:20.}) == 40., 'Occurrence-distance self check')
    base_test = CharDataset(CfgNode(**saved['data']), '\n\n'.join(r['rid_list'] for r in test), vocabulary)
    adapter = GravityTrajectoryDataset(base_test)  # Encoding only; NO gravity files/statistics/weights.
    windows = DeterministicTrajectoryWindows(adapter, range(len(test)))
    coverage = coverage_check(adapter, windows)
    overlap = sequence_overlap(train, test)
    overlap['test_sections_matching_audited_preference_sources'] = sum(
        any(i in preference_rows for i, r in enumerate(train) if r['traj_id'] in item['training_trajectory_ids'])
        for item in overlap['matches'])
    audit_result = {'passed': True, 'parent_leakage': False, 'all_original_training_parents': len(train_parents),
        'all_original_test_parents': len(test_parents), 'internal_training_parents': len(split['train_parents']),
        'internal_validation_parents': len(split['validation_parents']),
        'audited_possible_preference_parents': len(preference_parents),
        'train_test_parent_overlap': sorted(train_parents & test_parents),
        'test_preference_parent_overlap': sorted(test_parents & preference_parents),
        'exact_sequence_overlap': overlap, 'test_coverage': counts, 'count_discrepancies': [],
        'train_section_status_counts': train_status, 'test_section_status_counts': test_status,
        'export_verification': export, 'target_coverage': coverage,
        'elapsed_seconds': time.monotonic()-started,
        'interpretation': 'Independent parents, not guaranteed novel routes: exact duplicates are retained and reported.'}
    write_json(output/'overlap_coverage_audit.json', audit_result)
    write_json(output/'eligibility_manifest.json', manifest)
    protocol = {'phase': 'locked_before_gpu_evaluation', 'saved_config': saved,
        'resolved_selected_checkpoint': str(selected.resolve()), 'selected_update': SELECTED_UPDATE,
        'assessment': ASSESSMENT,
        'checkpoints': {k: fingerprint(p) for k,p in CHECKPOINTS.items()},
        'input_fingerprints': before, 'generation_seeds': list(SEEDS), 'prompt_lengths': list(PROMPTS),
        'temperature': 1., 'do_sample': True, 'top_k': None,
        'max_total_tokens_including_leading_eos': 81, 'max_road_tokens': 80,
        'eos_priority_at_length_cap': True, 'mode': 'eval/inference_mode',
        'per_case_seed': 'Existing stable_seed: SHA256([base_seed,str(trajectory_id),prompt_length])',
        'all_retained_sections': len(test), 'validation_generation_cache_used': False,
        'checkpoint_selection_locked': True, 'test_gravity_fitting': False,
        'test_loss': 'Unweighted token mean; existing deterministic windows; every target once including EOS',
        'distance': 'Meters summed for every occurrence, including supplied prompts and revisits',
        'bins': 'Shared road count width 1 and distance width 250 m spanning all outputs/references; sensitivity 100/250/500 m',
        'statistics': 'Paired descriptive results only; no independent-sample claims for siblings/seeds',
        'no_training_preferences_ppo_or_checkpoint_reselection': True}
    write_json(output/'locked_protocol.json', protocol)
    shutil.copy2(Path(__file__).resolve(), output/'evaluation_script.py')
    with (output/'checkout.diff').open('x') as stream:
        stream.write(git_record('diff', '--binary'))
    run_tests(output)
    assert_inputs(protocol)
    print(json.dumps({'audit_passed': True, 'output': str(output), 'test_sections': len(test),
                      'test_parents': len({r['parent_traj_id'] for r in test}),
                      'exact_duplicate_test_sections': overlap['test_sections_matching_training_sequences'],
                      'target_coverage': coverage}), flush=True)


def paired_metrics(records):
    lookup = {(r['checkpoint'], r['trajectory_id'], r['prompt_length'], r['seed']): r for r in records}
    output = {}
    for p in PROMPTS:
        output[str(p)] = {}
        for s in (*SEEDS, 'pooled'):
            baseline = [r for r in records if r['checkpoint'] == LABELS[0] and r['prompt_length'] == p
                        and (s == 'pooled' or r['seed'] == s)]
            pairs = [(r, lookup[(LABELS[1], r['trajectory_id'], p, r['seed'])]) for r in baseline]
            output[str(p)][str(s)] = {'pairs': len(pairs), 'direction': 'selected DPO minus pretrained',
                'identical_routes': sum(a['roads'] == b['roads'] for a,b in pairs),
                **{m+'_delta': describe([float(b[m])-float(a[m]) for a,b in pairs]) for m in
                   ('road_count','distance_m','immediate_stop','has_repeated_road','unique_road_fraction')}}
    return output


def make_plots(output, records, references, bins):
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, label in [('road_count','Full-route road count'), ('distance_m','Distance (m)')]:
        fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for row,p in enumerate(PROMPTS):
            groups = [('Test references', [r[metric] for r in references[p].values()], 'black')]
            groups += [(name, [r[metric] for r in records if r['checkpoint']==name and r['prompt_length']==p], color)
                       for name,color in zip(LABELS, ('#2864b0','#d56425'))]
            for name,values,color in groups:
                counts = np.histogram(values,bins[metric])[0]
                axes[row,0].stairs(counts/counts.sum(),bins[metric],label=name,color=color)
                values = np.sort(values)
                axes[row,1].step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name,color=color)
            for col in (0,1):
                axes[row,col].set_title(f'{p}-road prompt; all eligible test sections; three seeds')
                axes[row,col].set_xlabel(label)
                axes[row,col].legend(fontsize=8)
                axes[row,col].set_ylabel('Probability mass' if col==0 else 'Cumulative probability')
        for suffix in ('png','svg'):
            fig.savefig(output/f'{metric}_distributions.{suffix}',dpi=170)
        plt.close(fig)


def report(output, metrics, audit_result):
    overlap = audit_result['exact_sequence_overlap']
    lines = [f'# NYCAll test assessment: pretrained vs selected update-{SELECTED_UPDATE} DPO', '',
        ASSESSMENT + '. This previously examined test split is reproduction evidence, not newly untouched evidence.', '',
        f'Update {SELECTED_UPDATE} was selected by configuration before test evaluation. No other milestone was tested, '
        'no checkpoint was reselected or replaced, and no training, preferences or PPO was run.', '',
        '## Independence and coverage', '',
        f"All {audit_result['all_original_test_parents']:,} original test parents are disjoint from all "
        f"{audit_result['all_original_training_parents']:,} original training parents, including saved internal "
        f"validation and {audit_result['audited_possible_preference_parents']:,} audited possible DPO-source parents. "
        'Preference sources were re-derived from the fingerprinted artifact and checked against the previous audit.', '',
        f"Exact road-sequence overlap is separate: **{overlap['test_sections_matching_training_sequences']} test sections "
        f"from {overlap['test_parents_with_matching_sequences']} parents** match prepared training sequences; "
        f"{overlap['test_sections_matching_audited_preference_sources']} match audited DPO preference sources. "
        'All duplicates remain in the assessment. Parent independence does not establish route novelty.', '',
        f"Coverage matches preprocessing: {audit_result['test_coverage']}. "
        'This is prepared-network assessment, not coverage of excluded test trips. '
        'No count discrepancy was found. CSVs, provenance, original source hashes, vocabulary and directed relations passed checks.', '',
        '## Locked protocol', '',
        'Every retained section was considered. Each eligible prompt is sampled once for seeds 42/43/44 for '
        'both checkpoints with an identical SHA-256-derived per-case seed. Both are generated anew, sequentially '
        'on CUDA in eval/inference mode: temperature 1, sampling, no top-k, original corrected adjacency masking, '
        'no EOS manipulation/filtering/regeneration. The leading EOS counts toward 81 total tokens (at most 80 '
        'roads); EOS takes priority at the cap. Distances include every road occurrence and prompt, in meters.', '',
        '| Prompt | Eligible sections | Excluded, no continuation | Eligible parents | Reference roads | Reference meters | Reference revisit |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for p in PROMPTS:
        e, r = metrics['eligibility'][str(p)], metrics['reference_quality'][str(p)]
        lines.append(f"| {p} | {e['eligible']} | {e['ineligible']} | {e['parents']} | {r['road_count']['mean']:.3f} | {r['distance_m']['mean']:.1f} | {r['repeated_road_route_rate']:.2%} |")
    lines += ['', 'Three-road results exclude references with at most three roads, and therefore describe a longer-route '
              'subset. Loss still covers all sections and targets.', '', '## Pooled results', '',
        '| Checkpoint | Prompt | Roads | Meters | Count JS | Distance JS | Revisit | Unique-road fraction | Immediate stop | EOS | Cap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        for p in PROMPTS:
            g = metrics['aggregate'][name][str(p)]
            lines.append(f"| {name} | {p} | {g['road_count']['mean']:.3f} | {g['distance_m']['mean']:.1f} | {g['road_count_js_distance']:.4f} | {g['distance_m_js_distance']:.4f} | {g['repeated_road_route_rate']:.2%} | {g['per_route_unique_road_fraction']['mean']:.4f} | {g['immediate_stop_rate']:.2%} | {g['eos_rate']:.2%} | {g['length_cap_rate']:.3%} |")
    lines += ['', 'JS is base-2 Jensen–Shannon distance (smaller is closer), not divergence. Shared bins span '
        'both checkpoints and references; integer counts and 250 m distances, with 100/250/500 m sensitivity '
        'results in metrics. Median, percentiles, histograms and standard deviations are also saved.', '',
        '## Per-seed evidence', '',
        '| Checkpoint | Prompt | Seed | Count JS | Distance JS | Revisit | Immediate stop | Cap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        for p in PROMPTS:
            for s in SEEDS:
                g = metrics['per_seed'][name][str(p)][str(s)]
                lines.append(f"| {name} | {p} | {s} | {g['road_count_js_distance']:.4f} | {g['distance_m_js_distance']:.4f} | {g['repeated_road_route_rate']:.2%} | {g['immediate_stop_rate']:.2%} | {g['length_cap_rate']:.3%} |")
    lines += ['', '## Validity, diversity and token loss', '',
        '| Checkpoint | Prompt | Valid road IDs | Connected actual pairs | Invalid-route fraction | Single-road samples | Unique routes / samples | Distinct origins |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in LABELS:
        for p in PROMPTS:
            g = metrics['aggregate'][name][str(p)]
            lines.append(f"| {name} | {p} | {g['valid_road_id_token_rate']:.2%} | {g['adjacent_pairs']-g['invalid_transitions']}/{g['adjacent_pairs']} | {g['fraction_samples_with_invalid_transition']:.2%} | {g['single_road_samples']} | {g['exact_unique_routes']}/{g['samples']} | {g['distinct_origins']} |")
    lines += ['', 'Connectivity counts actual adjacent pairs; single-road samples do not inflate it. Exact-route '
        'uniqueness also reflects supplied-prompt diversity. Revisit statistics include repeats already in prompts.', '',
        '| Checkpoint | Unweighted test token loss | Targets | Windows |', '|---|---:|---:|---:|']
    for name,g in metrics['test_token_loss'].items():
        lines.append(f"| {name} | {g['loss']:.9f} | {g['tokens']} | {g['windows']} |")
    lines += ['', 'Token mean uses the unchanged deterministic evaluation implementation, with all targets covered '
        'once, including first-road and terminal-EOS prediction; padding is ignored. No gravity statistics '
        'were fitted or used. This measures language-model behavior, not an exact-route prediction score.', '',
        '## Final interpretation', '']
    for p in PROMPTS:
        a,b = [metrics['aggregate'][name][str(p)] for name in LABELS]
        improved = all(b[k] < a[k] for k in ('road_count_js_distance','distance_m_js_distance'))
        lines.append(f"For {p}-road prompts, both pooled distribution distances {'improve' if improved else 'do not both improve'} "
            f"at update {SELECTED_UPDATE}: count {a['road_count_js_distance']:.4f} → {b['road_count_js_distance']:.4f}, "
            f"distance {a['distance_m_js_distance']:.4f} → {b['distance_m_js_distance']:.4f}; revisiting "
            f"{a['repeated_road_route_rate']:.2%} → {b['repeated_road_route_rate']:.2%} "
            f"({100*(b['repeated_road_route_rate']-a['repeated_road_route_rate']):+.2f} percentage points).")
    lines += ['', 'These are paired descriptive results, not independent observations across repeated seeds or '
        'sibling sections. No confidence intervals or significance claims are made. Seed variation and paired '
        'differences are saved. Different plausible continuations need not reproduce the exact recorded trip; '
        'longer routes alone are not success. Distribution improvement must be balanced against additional '
        'revisiting and the much lower real-route revisit rate. No single winning score is manufactured, '
        'no checkpoint is automatically replaced, and test results cannot authorize milestone retuning. '
        '**PPO remains out of scope.**', '', '## Verification and artifacts', '',
        f"Strict loads passed for both checkpoints. {metrics['verification']['actual_samples']:,} expected paired "
        f"generations were retained with no failures/drops. All {metrics['verification']['probability_vectors_checked']:,} "
        'probability vectors were finite/normalized and assigned exactly zero forbidden-transition probability. '
        'Input hashes and the selected symlink are unchanged. Regression logs and software/hardware details are saved.', '',
        '[Metrics](metrics.json) · [Raw generations](generated_trajectories.jsonl) · '
        '[Eligibility](eligibility_manifest.json) · [Independence/coverage audit](overlap_coverage_audit.json) · '
        '[Locked protocol and hashes](locked_protocol.json) · [Environment/timing](environment_timing.json) · '
        '[Tests](test_results.json) · [Count plot](road_count_distributions.png) · '
        '[Distance plot](distance_m_distributions.png)', '', '### Compared checkpoint hashes', '']
    protocol = json.loads((output/'locked_protocol.json').read_text())
    for name,info in protocol['checkpoints'].items():
        lines.append(f"- {name}: `{info['sha256']}` (`{info['path']}`)")
    with (output/'report.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def evaluate(output):
    output = evaluation_output(output)
    protocol = json.loads((output/'locked_protocol.json').read_text())
    require(not (output/'generated_trajectories.jsonl').exists(), 'Refusing partial-run resume or overwrite')
    require(not (output/'metrics.json').exists(), 'Refusing result overwrite')
    require(all(r['exit_code']==0 for r in json.loads((output/'test_results.json').read_text())), 'Tests failed')
    assert_inputs(protocol)
    require(torch.cuda.is_available(), 'CUDA required; request GPU sandbox access')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'numpy': np.__version__, 'gpu': torch.cuda.get_device_name(),
        'gpu_free_and_total_bytes_before': list(torch.cuda.mem_get_info()), 'pid': os.getpid(),
        'torch_threads': torch.get_num_threads(), 'tf32': False,
        'nvidia_smi': subprocess.run(['nvidia-smi'],capture_output=True,text=True).stdout,
        'checkpoint_timings': {}}
    with (output/'environment_before.json').open('x') as stream:
        json.dump(environment, stream, indent=2)
    manifest = json.loads((output/'eligibility_manifest.json').read_text())
    audit_result = json.loads((output/'overlap_coverage_audit.json').read_text())
    saved = protocol['saved_config']
    geo, vocabulary, _, relations = load_geo_data('NYCAll', DATA.parent)
    lengths = dict(zip(map(int,vocabulary), map(float,geo['length'])))
    require(all(math.isfinite(v) and v >= 0 for v in lengths.values()), 'Invalid road length')
    edges = {tuple(map(int,pair)) for pair in relations}
    rows = read_rows(DATA/'NYCAll_Taxi_trajectory_test.csv')
    require(manifest == eligibility(rows), 'Eligibility manifest changed since audit')
    base = CharDataset(CfgNode(**saved['data']), '\n\n'.join(r['rid_list'] for r in rows), vocabulary)
    adapter = GravityTrajectoryDataset(base)
    windows = DeterministicTrajectoryWindows(adapter, range(len(rows)))
    require(coverage_check(adapter, windows) == audit_result['target_coverage'], 'Target coverage changed')
    references = {p:{r['trajectory_id']:{'roads':r['reference_roads'],
        'prompt_roads':r['reference_roads'][:p], 'immediate_stop':False, 'termination':'reference',
        **route_quality(r['reference_roads'], lengths, edges)} for r in manifest if p in r['eligible_prompt_lengths']}
        for p in PROMPTS}
    adjacency = od_pair_to_adjacency_matrix(relations,'cuda')
    require(tuple(adjacency.shape) == (len(vocabulary)+1,)*2, 'Adjacency shape mismatch')
    records, losses, strict_loads = [], {}, {}
    with (output/'progress.jsonl').open('x', buffering=1) as progress, \
         (output/'generated_trajectories.jsonl').open('x', buffering=1) as raw, \
         (output/'failed_cases.jsonl').open('x', buffering=1) as failures:
        def log(event):
            event['elapsed_seconds'] = round(time.monotonic()-started,3)
            progress.write(json.dumps(event,allow_nan=False)+'\n')
            print(json.dumps(event,allow_nan=False),flush=True)
        for name,path in CHECKPOINTS.items():
            checkpoint_start = time.monotonic()
            torch.manual_seed(0)
            model = GPT(CfgNode(**saved['model']), adj_matrix=adjacency).cuda()
            state = torch.load(path, map_location='cpu', weights_only=True)
            require(all(torch.isfinite(v).all().item() for v in state.values()), 'Nonfinite checkpoint')
            loading = model.load_state_dict(state, strict=True)
            strict_loads[name] = {'passed': not loading.missing_keys and not loading.unexpected_keys,
                                  'tensors':len(state)}
            del state
            model.eval()
            config = CfgNode(**saved['training']); config.device = 'cuda'
            trainer = Trainer(config,model,adapter,[],[],val_dataset=windows)
            loss_start = time.monotonic()
            with torch.inference_mode():
                loss = trainer.evaluate()
            require(trainer.val_tokens == audit_result['target_coverage']['targets'], 'Loss target count mismatch')
            losses[name] = {'loss': loss, 'tokens':trainer.val_tokens, 'windows':len(windows),
                            'elapsed_seconds':time.monotonic()-loss_start, 'unweighted':True}
            log({'checkpoint':name,'test_token_loss':loss,'targets':trainer.val_tokens})
            with torch.inference_mode():
                for p in PROMPTS:
                    eligible = [r for r in manifest if p in r['eligible_prompt_lengths']]
                    for seed in SEEDS:
                        for i,item in enumerate(eligible):
                            prompt = item['reference_roads'][:p]
                            derived = stable_seed(seed,item['trajectory_id'],p)
                            identity = {'checkpoint':name,'trajectory_index':item['trajectory_index'],
                                'trajectory_id':item['trajectory_id'],'parent_id':item['parent_id'],
                                'prompt_length':p,'prompt_roads':prompt,'seed':seed,'derived_seed':derived}
                            try:
                                tokens = [base.stoi[base.EOS_TOKEN],*(base.stoi[str(r)] for r in prompt)]
                                ids,termination,steps,error = generate_route(model,
                                    torch.tensor([tokens],device='cuda'),base.stoi[base.EOS_TOKEN],derived)
                                roads = [int(base.itos[token]) for token in ids]
                                record = {**identity,'roads':roads,'termination':termination,
                                    'immediate_stop':len(roads)==p, 'probability_steps':steps,
                                    'max_probability_normalization_error':error,
                                    'generation_origin':'new_cuda_test_generation',
                                    **route_quality(roads,lengths,edges)}
                                records.append(record)
                                raw.write(json.dumps(record,allow_nan=False)+'\n')
                            except Exception:
                                failures.write(json.dumps({**identity,'traceback':traceback.format_exc()})+'\n')
                                log({'failed_case':identity,'status':'stopped_no_silent_drops'})
                                raise
                            if (i+1)%100 == 0 or i+1 == len(eligible):
                                log({'checkpoint':name,'prompt':p,'seed':seed,'completed':i+1,'eligible':len(eligible)})
            environment['checkpoint_timings'][name] = {'elapsed_seconds':time.monotonic()-checkpoint_start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
            del trainer,model
            torch.cuda.empty_cache()
        verification = verify_pairing(records,manifest,checkpoints=LABELS)
        assert_inputs(protocol)
        verification.update(strict_loads=strict_loads,input_fingerprints_unchanged=True,
            selected_symlink_unchanged=True, finite_normalized_probabilities=True,forbidden_probabilities_zero=True,
            probability_vectors_checked=sum(r['probability_steps'] for r in records),
            max_probability_normalization_error=max(r['max_probability_normalization_error'] for r in records),
            failed_cases=0, target_coverage=audit_result['target_coverage'])
        all_values = records + [r for group in references.values() for r in group.values()]
        bins = {'road_count':np.arange(-.5,max(r['road_count'] for r in all_values)+1.5,1).tolist(),
            'distance_m':np.arange(0,math.ceil(max(r['distance_m'] for r in all_values)/250)*250+250,250).tolist()}
        metrics = {'aggregate':{},'per_seed':{},'seed_variation':{},'histogram_bins':bins,
            'test_token_loss':losses,'verification':verification,'paired':paired_metrics(records),
            'reference_quality':{},'eligibility':{}, 'assessment_only_no_reselection':True}
        for name in LABELS:
            metrics['aggregate'][name],metrics['per_seed'][name],metrics['seed_variation'][name] = {},{},{}
            for p in PROMPTS:
                subset = [r for r in records if r['checkpoint']==name and r['prompt_length']==p]
                refs = list(references[p].values())
                pooled = summarize(subset,refs,bins)
                pooled['eos_rate'] = pooled['termination_counts'].get('eos',0)/pooled['samples']
                metrics['aggregate'][name][str(p)] = pooled
                each = {str(s):summarize([r for r in subset if r['seed']==s],refs,bins) for s in SEEDS}
                for g in each.values():
                    g['eos_rate'] = g['termination_counts'].get('eos',0)/g['samples']
                metrics['per_seed'][name][str(p)] = each
                metrics['seed_variation'][name][str(p)] = {k:describe([g[k] for g in each.values()]) for k in
                    ('road_count_js_distance','distance_m_js_distance','repeated_road_route_rate','immediate_stop_rate')}
        for p in PROMPTS:
            refs = list(references[p].values())
            metrics['reference_quality'][str(p)] = summarize(refs,refs,bins)
            metrics['eligibility'][str(p)] = {'eligible':len(refs),'ineligible':len(manifest)-len(refs),
                'parents':len({r['parent_id'] for r in manifest if p in r['eligible_prompt_lengths']})}
        write_json(output/'metrics.json',metrics)
        make_plots(output,records,references,bins)
        environment['total_evaluation_seconds'] = time.monotonic()-started
        write_json(output/'environment_timing.json',environment)
        report(output,metrics,audit_result)
        log({'status':'completed','records':len(records),'output':str(output)})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('audit','evaluate'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-dir','--pipeline-run',dest='pipeline_run',type=Path,required=True,
                        help='Saved pipeline run; all checkpoint/data paths are run-local')
    parser.add_argument('--raw-data-dir',type=Path,required=True,help='Explicit original raw inputs')
    args = parser.parse_args(argv)
    configure_pipeline(args.pipeline_run, args.raw_data_dir)
    output = evaluation_output(args.output)
    try:
        (audit if args.phase=='audit' else evaluate)(output)
    except Exception:
        if output.is_dir() and not (output/f'{args.phase}_failure.json').exists():
            write_json(output/f'{args.phase}_failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
