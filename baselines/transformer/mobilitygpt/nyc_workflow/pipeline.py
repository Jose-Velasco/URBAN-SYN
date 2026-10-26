"""Run the established NYCAll recipe unattended, from raw inputs through both metric reports."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
import secrets

from .runtime import (PACKAGE, module_command, source_entries, snapshot_sources,
                      git_record, require_output)
from .preflight import MISSING_NETWORK, validate_configuration
STAGES = ('preflight', 'prepare', 'gravity', 'verify', 'safety', 'pretrain', 'preferences',
          'dpo', 'select', 'evaluation_audit', 'evaluate', 'canonical_metrics', 'report')
CAVEATS = [
    'Preferences sample the complete prepared training text, including internal pretraining validation.',
    'Original generator remains in training/dropout mode, with its uncapped generation behavior.',
    'Outer prompt is duplicated inside the scored completion; chosen/rejected EOS treatment is asymmetric.',
    'Original context truncation and road-ID-1 padding with zero completion masks are preserved.',
    'Parent separation does not imply route novelty; exact train/test route overlap is retained and counted.',
    'Evaluation covers retained training-network test sections, with explicit prompt eligibility exclusions.',
    'The 80-road generation cap cannot reproduce longer reference tails; revisiting remains a limitation.',
    'The previously examined test split is a reproduction assessment, not newly untouched evidence.',
]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Atomic journal updates; stage implementations retain exclusive output creation."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_info(path):
    path = Path(path)
    value = {'sha256': sha(path), 'bytes': path.stat().st_size}
    if path.is_symlink():
        value['symlink'] = os.readlink(path)
    return value


def inventory(paths):
    files = set()
    for path in paths:
        path = Path(path)
        require(path.exists(), f'Missing stage artifact: {path}')
        files.update(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else files.add(path)
    return {str(p): file_info(p) for p in sorted(files)}


def validate_inventory(saved):
    for path, expected in saved.items():
        require(Path(path).is_file() and file_info(path) == expected, f'Artifact/input changed: {path}')


def source_paths():
    return sorted(source_entries().values())


def dependencies():
    return dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                       if d.metadata['Name']))


def configuration(input_dir, smoke=False, canonical_network=None, network_layer='roads',
                  network_mode_column=None, metric_jobs=6):
    return {'schema_version': 3, 'input_dir': str(Path(input_dir).resolve()), 'dataset': 'NYCAll',
        'assessment': 'Integration diagnostics only' if smoke else 'Locked NYCAll reproduction assessment',
        'smoke': smoke, 'seed': 160,
        'preprocessing': {'candidate_road_limit': 50000,
                          'policy': 'Existing conservative unique-two-hop repair; whole sections; training-derived observed network'},
        'pretraining': {'updates': 2 if smoke else 3000, 'safety_updates': 50,
            'batch_size': 64, 'learning_rate': 1e-5, 'workers': 0,
            'eval_every': 250, 'sample_count': 32 if smoke else 500,
            'gravity': True, 'validation_split': 0.2, 'split_seed': 42,
            'adjacency': True, 'adjacency_dtype': 'torch.bool'},
        'dpo': {'preferences': 8 if smoke else 1000, 'updates': 2 if smoke else 250,
            'milestones': [1, 2] if smoke else [50, 100, 250], 'selected_update': 1 if smoke else 100,
            'batch_size': 8, 'learning_rate': 1e-5, 'beta': 0.1,
            'trainable_policy': 'Existing CLI default: LoRA plus vocabulary output head'},
        'evaluation': {'all_eligible_retained_test_sections': True, 'prompt_roads': [1, 3],
            'seeds': [42, 43, 44], 'temperature': 1.0, 'top_k': None,
            'max_total_tokens': 81, 'generation': 'Existing paired per-case SHA256 seed protocol'},
        'metrics': {'canonical_network': str(Path(canonical_network).resolve()) if canonical_network is not None else None,
            'network_layer': network_layer, 'network_mode_column': network_mode_column, 'jobs': metric_jobs,
            'missing_network_policy': 'Complete 11 metrics; off-network unavailable; canonical assessment incomplete',
            'definitions': 'Raw-reference notebook-compatible count mixing; FastDTW radius 1'},
        'accepted_limitations': CAVEATS}


def default_output():
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+secrets.token_hex(3)
    return PACKAGE/'output'/run_id


def commands(run, config):
    modules = {'pretraining': 'pretraining', 'pipeline': 'pipeline',
               'dpo': 'dpo', 'evaluation': 'evaluation',
               'preprocessing': 'preprocessing', 'metrics': 'metrics'}
    def script(name, *args):
        return module_command('nyc_workflow.'+modules[name], *args)
    prepared, p, d = run/'prepared', config['pretraining'], config['dpo']
    def pretrain(output, steps):
        return script('pretraining', '--dataset', 'NYCAll', '--data-root', prepared,
            '--output', run/output, '--steps', steps, '--batch-size', p['batch_size'],
            '--learning-rate', p['learning_rate'], '--workers', p['workers'],
            '--eval-every', p['eval_every'], '--sample-count', p['sample_count'])
    def worker(stage):
        return script('pipeline', '--_worker', stage, '--input-dir', config['input_dir'], '--output-dir', run)
    dpo_base = ['child', '--experiment-dir', run, '--data-root', prepared,
                '--pretrained', run/'pretrain/model_best.pt', '--samples', d['preferences'],
                '--updates', d['updates']]
    def evaluator(phase):
        return script('evaluation', phase, '--pipeline-run', run, '--raw-data-dir', config['input_dir'], '--output', run/'evaluation')
    m = config['metrics']
    metric_arguments = ['--run-dir', run, '--raw-data-dir', config['input_dir'],
        '--output-dir', run/'canonical_metrics', '--jobs', m['jobs']]
    if m['canonical_network'] is None:
        metric_arguments += ['--allow-missing-network']
    else:
        metric_arguments += ['--canonical-network', m['canonical_network'], '--network-layer', m['network_layer']]
        if m['network_mode_column'] is not None:
            metric_arguments += ['--network-mode-column', m['network_mode_column']]
    return {
        'preflight': worker('preflight'),
        'prepare': script('preprocessing', '--source-dir', config['input_dir'],
            '--output-root', prepared, '--dataset-name', 'NYCAll',
            '--candidate-road-limit', config['preprocessing']['candidate_road_limit']),
        'gravity': worker('gravity'), 'verify': worker('verify'),
        'safety': pretrain('safety', p['safety_updates']),
        'pretrain': pretrain('pretrain', p['updates']) + ['--require-short-check', str(run/'safety')],
        'preferences': script('dpo', *dpo_base, '--action', 'generate', '--name', 'preferences'),
        'dpo': script('dpo', *dpo_base, '--action', 'train', '--name', 'training',
            '--artifact', run/'preferences/preference_dataset_dpo', '--checkpoint-updates', *d['milestones']),
        'select': worker('select'), 'evaluation_audit': evaluator('audit'),
        'evaluate': evaluator('evaluate'), 'canonical_metrics': script('metrics', *metric_arguments),
        'report': worker('report')}


def outputs(run, stage):
    data = run/'prepared/NYCAll-Taxi'
    return {
        'preflight': [run/'preflight.json'],
        'prepare': [data, run/'prepared/Trajs_NYCAll.txt'],
        'gravity': [data/'regions', data/'NYCAll_Taxi_trajectory_train_w_gravity.csv',
                    data/'NYCAll_Taxi_trajectory_train_w_gravity.metadata.json'],
        'verify': [run/'verification.json', run/'verified_split.json'],
        'safety': [run/'safety'], 'pretrain': [run/'pretrain'],
        'preferences': [run/'preferences'], 'dpo': [run/'training'],
        'select': [run/'selection.json', run/'model_selected.pt'],
        'evaluation_audit': [run/'evaluation'], 'evaluate': [run/'evaluation'],
        'canonical_metrics': [run/'canonical_metrics'],
        'report': [run/'report.json', run/'report.md', run/'metrics.json', run/'report.html'],
    }[stage]


def check_stage(run, stage, config):
    """Reject successful process exits with incomplete or invalid scientific outputs."""
    if stage == 'preflight':
        require(read(run/'preflight.json')['passed'], 'Preflight validation failed')
    elif stage == 'prepare':
        from nyc_workflow.preprocessing import verify_export
        result = verify_export(run/'prepared/NYCAll-Taxi', run/'prepared/Trajs_NYCAll.txt', 'NYCAll')
        require(result['passed'], 'Invalid prepared export')
    elif stage == 'gravity':
        require(read(outputs(run, stage)[2])['statistics_rows'] > 0, 'Empty gravity fit')
    elif stage == 'verify':
        result = read(run/'verification.json')
        require(result['passed'] and result['parent_overlap'] == 0 and result['adjacency']['dtype'] == 'torch.bool',
                'Pretraining verification failed')
    elif stage in ('safety', 'pretrain'):
        result = read(run/stage/'results.json')
        expected = config['pretraining']['safety_updates' if stage == 'safety' else 'updates']
        require(result['passed'] and result['steps'] == result['finite_gradient_steps'] == expected
                and result['memory_budget_passed'] and result['adjacency_dtype'] == 'torch.bool'
                and result['checkpoint_reload'] == 'strict state dict and exact logits passed',
                f'Invalid {stage} result')
        # Best is the downstream initialization, so check it as well as the final checkpoint.
        import torch
        from mobilitygpt.config_utils import CfgNode
        from mobilitygpt.model import GPT
        model = GPT(CfgNode(**read(run/stage/'config.json')['model']))
        for name in ('model_best.pt', 'model_pretrain.pt'):
            state = torch.load(run/stage/name, map_location='cpu', weights_only=True)
            require(all(torch.isfinite(v).all().item() for v in state.values()), 'Nonfinite checkpoint')
            model.load_state_dict(state, strict=True)
    elif stage in ('preferences', 'dpo'):
        directory = run/('preferences' if stage == 'preferences' else 'training')
        result = read(directory/'result.json')
        require(result['status'] == 'passed' and result['policy_initial_strict_load'], 'DPO stage failed')
        require(read(directory/'dataset_audit.json')['examples'] == config['dpo']['preferences'], 'Preference count mismatch')
        if stage == 'dpo':
            require(result['updates_completed'] == config['dpo']['updates'] and result['reference_unchanged']
                and result['reference_gradients_absent'] and result['policy_parameters_finite']
                and not result['frozen_parameter_changes'] and result['changed_parameter_names']
                and result['same_shared_adjacency'] and result['reference_initial_strict_load'], 'Invalid DPO training')
            for step in config['dpo']['milestones']:
                require(result['milestone_checkpoints'][str(step)]['strict_cpu_loading'] == '<All keys matched successfully>',
                        'Milestone strict reload failed')
    elif stage == 'select':
        require(read(run/'selection.json')['selected_update'] == config['dpo']['selected_update'], 'Selection changed')
    elif stage == 'evaluation_audit':
        require(read(run/'evaluation/overlap_coverage_audit.json')['passed'], 'Evaluation audit failed')
        require(all(r['exit_code'] == 0 for r in read(run/'evaluation/test_results.json')), 'Regression failure')
    elif stage == 'evaluate':
        result = read(run/'evaluation/metrics.json')['verification']
        require(result['actual_samples'] == result['expected_samples'] and result['failed_cases'] == 0
            and result['finite_normalized_probabilities'] and result['forbidden_probabilities_zero']
            and result['target_coverage']['each_target_position_covered_exactly_once']
            and all(v['passed'] for v in result['strict_loads'].values()), 'Incomplete paired evaluation')
    elif stage == 'canonical_metrics':
        from .metrics import METRICS
        import math
        result = read(run/'canonical_metrics/results.json')
        network = config['metrics']['canonical_network'] is not None
        require(result['input_hashes_preserved'] and len(result['results']) == 12,
                'Incomplete canonical metric groups or changed inputs')
        require(result['complete'] == network and result['status'] == ('complete' if network else 'incomplete'),
                'Canonical assessment status does not match network availability')
        for row in result['results']:
            for metric in METRICS:
                if metric == 'off_network_rate' and not network:
                    require(row[metric] is None and row['unavailable_reasons'].get(metric), 'Fabricated off-network score')
                else:
                    require(isinstance(row[metric], (int, float)) and math.isfinite(row[metric]), f'Unavailable/nonfinite metric: {metric}')
    elif stage == 'report':
        require(read(run/'report.json')['passed'], 'Report did not pass')
        require((run/'report.html').is_file(), 'Offline HTML report missing')


def worker(stage, run):
    run = require_output(run)
    config = read(run/'pipeline_config.json')
    if stage == 'preflight':
        from .preflight import validate
        save(run/'preflight.json', validate(config))
    elif stage == 'gravity':
        from preprocessing.gravity_based_sampling import GravityBasedSampler
        GravityBasedSampler('NYCAll', data_root=run/'prepared').process()
    elif stage == 'verify':
        from nyc_workflow.preprocessing import verify_export, measure_adjacency
        from nyc_workflow.pretraining import prepare_data, build_config, parse_args
        args = parse_args(['--dataset', 'NYCAll', '--data-root', str(run/'prepared'),
                           '--batch-size', '64', '--learning-rate', '1e-5', '--workers', '0'])
        _, _, _, _, _, diagnostics, split = prepare_data(build_config(args))
        export = verify_export(run/'prepared/NYCAll-Taxi', run/'prepared/Trajs_NYCAll.txt', 'NYCAll')
        adjacency = measure_adjacency(run/'prepared', 'NYCAll', 'cuda')
        save(run/'verification.json', {**diagnostics, 'passed': True, 'export': export, 'adjacency': adjacency})
        save(run/'verified_split.json', split)
    elif stage == 'select':
        from nyc_workflow.paired import fingerprint
        step = config['dpo']['selected_update']
        target = run/f'training/model_step_{step:06d}.pt'
        require(target.is_file(), 'Configured selected milestone missing')
        (run/'model_selected.pt').symlink_to(target.relative_to(run))
        save(run/'selection.json', {'selected_update': step,
            'basis': 'Pipeline configuration locked before training or test evaluation; no metric selection',
            'checkpoint': fingerprint(run/'model_selected.pt')})
    elif stage == 'report':
        from .reporting import write_report
        write_report(run, config, CAVEATS)


def run_pipeline(args):
    require(not args.resume or args.output_dir is not None, '--resume requires the original explicit --output-dir')
    network = getattr(args, 'canonical_network', None)
    run = require_output(args.output_dir if args.output_dir is not None else default_output(),
                         [args.input_dir] + ([network] if network is not None else []))
    config = configuration(args.input_dir, args.smoke, network,
        getattr(args, 'network_layer', 'roads'), getattr(args, 'network_mode_column', None),
        getattr(args, 'metric_jobs', 6))
    validate_configuration(config)
    print(json.dumps({'output': str(run), 'smoke': args.smoke}), flush=True)
    print(MISSING_NETWORK if network is None else f'Canonical network will be validated before training: {network}', flush=True)
    input_paths = [Path(config['input_dir'])/name for name in
                   ('nyc.geo', 'nyc.rel', 'nyc_mm_train.csv', 'nyc_mm_test.csv')]
    if network is not None:
        input_paths.append(Path(config['metrics']['canonical_network']))
    inputs = inventory(input_paths)
    if args.resume:
        require(run.is_dir(), 'Resume directory does not exist')
    else:
        run.mkdir(parents=True, exist_ok=False)
    with (run/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = inventory(source_paths())
        identity = {'configuration': config, 'inputs': inputs, 'sources': sources,
                    'dependencies': dependencies(), 'python': sys.version,
                    'executable': str(Path(sys.executable).absolute())}
        if args.resume:
            require(read(run/'identity.json') == identity, 'Restart refused: input/configuration/code/dependency mismatch')
            states = read(run/'stage_status.json')
            for stage, record in states.items():
                require(record['status'] == 'complete', f'Restart refused: {stage} is partial/failed; use a fresh directory')
                validate_inventory(record['outputs'])
        else:
            save(run/'pipeline_config.json', config)
            save(run/'identity.json', identity)
            save(run/'environment.json', {'platform': platform.platform(), **identity})
            with (run/'requirements-freeze.txt').open('x') as stream:
                subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
            snapshot_sources(run/'source')
            for name, arguments in [('checkout.diff', ['diff', '--binary']),
                                    ('git_status.txt', ['status', '--short'])]:
                with (run/name).open('x') as stream:
                    stream.write(git_record(*arguments))
            (run/'logs').mkdir()
            states = {}
            save(run/'stage_status.json', states)
        planned = commands(run, config)
        for stage in STAGES:
            if stage in states:
                require(states[stage]['command'] == planned[stage], f'Command changed: {stage}')
                print(json.dumps({'stage': stage, 'status': 'verified_complete_skip'}), flush=True)
            else:
                validate_inventory(inputs)
                validate_inventory(sources)
                for previous in states.values():
                    validate_inventory(previous['outputs'])
                command = planned[stage]
                record = {'status': 'running', 'command': command, 'source_package': str(PACKAGE),
                          'started_unix': time.time(), 'input_identity_sha256': sha(run/'identity.json'),
                          'upstream_output_hashes': {s: hashlib.sha256(json.dumps(r['outputs'], sort_keys=True).encode()).hexdigest()
                                                     for s, r in states.items()}}
                states[stage] = record
                save(run/'stage_status.json', states)
                print(json.dumps({'stage': stage, 'status': 'running', 'command': command}), flush=True)
                started = time.monotonic()
                try:
                    with (run/'logs'/f'{stage}.log').open('x') as stream:
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                        _, status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(status)
                    record.update(exit_code=process.returncode, elapsed_seconds=time.monotonic()-started,
                        peak_host_ram_bytes=usage.ru_maxrss*1024, user_cpu_seconds=usage.ru_utime,
                        system_cpu_seconds=usage.ru_stime)
                    require(process.returncode == 0, f'{stage} exited {process.returncode}; see logs/{stage}.log')
                    check_stage(run, stage, config)
                    validate_inventory(inputs)
                    record.update(status='complete', outputs=inventory(outputs(run, stage)))
                    # GPU peaks are measured within each CUDA stage, in its saved result/environment.
                    memory_paths = {'safety': run/'safety/results.json', 'pretrain': run/'pretrain/results.json',
                        'preferences': run/'preferences/result.json', 'dpo': run/'training/result.json',
                        'verify': run/'verification.json', 'evaluate': run/'evaluation/environment_timing.json'}
                    if stage in memory_paths:
                        record['memory_and_timing_artifact'] = str(memory_paths[stage])
                except BaseException:
                    record.update(status='failed', elapsed_seconds=time.monotonic()-started, traceback=traceback.format_exc())
                    save(run/'stage_status.json', states)
                    raise
                save(run/'stage_status.json', states)
                print(json.dumps({'stage': stage, 'status': 'complete', 'seconds': record['elapsed_seconds']}), flush=True)
            if args.stop_after == stage:
                return
        validate_inventory(inputs)
        validate_inventory(sources)
        if not (run/'artifact_manifest.json').exists():
            save(run/'artifact_manifest.json', inventory([run/name for name in
                ('prepared', 'safety', 'pretrain', 'preferences', 'training', 'evaluation', 'canonical_metrics', 'source', 'logs',
                 'pipeline_config.json', 'identity.json', 'environment.json', 'requirements-freeze.txt',
                 'checkout.diff', 'git_status.txt', 'selection.json', 'model_selected.pt', 'verification.json',
                 'verified_split.json', 'preflight.json', 'stage_status.json', 'report.json', 'report.md', 'report.html', 'metrics.json')]))
        validate_inventory(read(run/'artifact_manifest.json'))
        print(json.dumps({'status': 'passed', 'assessment': config['assessment'], 'output': str(run),
            'canonical_assessment': 'complete' if network is not None else 'incomplete',
            'limitation': None if network is not None else MISSING_NETWORK}), flush=True)
        print(f'Open results: {(run/"report.html").resolve()}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Fresh run directory; default nyc_workflow/output/<UTC timestamp>-<nonce>')
    parser.add_argument('--canonical-network', type=Path, help='Optional canonical road network; validated before training')
    parser.add_argument('--network-layer', default='roads')
    parser.add_argument('--network-mode-column')
    parser.add_argument('--metric-jobs', type=int, default=6, help='Bounded CPU workers for notebook-compatible metrics')
    parser.add_argument('--smoke', action='store_true', help='Fewer updates/preferences; full data and test coverage; integration diagnostics only')
    parser.add_argument('--resume', action='store_true', help='Verify and skip completed matching stages; refuse partial stages')
    parser.add_argument('--stop-after', choices=STAGES, help='Stop cleanly at a stage boundary for later --resume')
    parser.add_argument('--_worker', choices=('preflight', 'gravity', 'verify', 'select', 'report'), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        require(args.output_dir is not None, 'Stage workers require an explicit run directory')
        worker(args._worker, args.output_dir.resolve())
    else:
        run_pipeline(args)


if __name__ == '__main__':
    main()
