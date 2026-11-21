"""Controlled, full-parameter NYC pretraining; never starts later training stages."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import resource
import subprocess
import time

import torch
from torch.utils.data import Dataset

from datasets.char_dataset import CharDataset
from datasets.data_utils import load_geo_data, load_trajectory_data
from nyc_workflow.data import (
    DeterministicTrajectoryWindows, GravityTrajectoryDataset, load_gravity_data,
    split_trajectory_indices,
)
from mobilitygpt.trainer_utils import initialize_model
from mobilitygpt.config import get_base_config
from mobilitygpt.config_utils import set_seed
from mobilitygpt.model import GPT
from mobilitygpt.trainer import Trainer
from mobilitygpt.trainer_utils import create_gravity_samplers
from .runtime import require_output, source_entries, snapshot_sources, git_record


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def memory_budget(free_bytes, total_bytes):
    return min(int(0.85 * total_bytes), free_bytes - 512 * 1024**2)


class MemoryBudgetExceeded(RuntimeError):
    pass


class ExposureDataset(Dataset):
    """Record consumed trajectory rows and actual window road exposure (workers=0)."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.draws = []
        self.batch_ends = []
        self.window_roads = set()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        self.draws.append((int(index), int(y.ne(-1).sum())))
        self.window_roads.update(t for t in set(x.tolist() + y.tolist())
                                 if t >= 0 and t != self.dataset.eos_id)
        return x, y


def check_short_compatibility(config, diagnostics, short, previous):
    if not short.get('passed') or short.get('steps') != 50 or short.get('profile_only'):
        raise ValueError('The complete 50-step check did not pass')
    if not short.get('memory_budget_passed'):
        raise ValueError('Short check did not pass the memory budget')
    if previous['data'] != config.data.to_dict():
        raise ValueError('Short-check dataset/settings differ')
    if previous['model'] != config.model.to_dict():
        raise ValueError('Short-check model settings differ')
    if previous['system']['seed'] != config.system.seed:
        raise ValueError('Short-check initialization seed differs')
    for key in ('gravity_sha256', 'source_sha256', 'geography_sha256', 'relations_sha256', 'text_sha256'):
        if short['gravity_diagnostics'].get(key) != diagnostics[key]:
            raise ValueError(f'Short-check data differs: {key}')
    current = json.loads(json.dumps(config.training.to_dict()))
    for key, value in current.items():
        if key != 'max_iters' and previous['training'].get(key) != value:
            raise ValueError(f'Short-check training setting differs: {key}')


def prepare_data(config):
    root = Path(config.data.root)
    _, vocabulary, _, relations = load_geo_data(config.data.dataset, root)
    base = CharDataset(config.data, load_trajectory_data(config), vocabulary)
    dataset = GravityTrajectoryDataset(base)
    directory = root / f"{config.data.dataset}-Taxi"
    source = directory / f"{config.data.dataset}_Taxi_trajectory_train.csv"
    gravity = directory / f"{config.data.dataset}_Taxi_trajectory_train_w_gravity.csv"
    raw, parents = load_gravity_data(dataset, gravity, source)
    if not len(dataset) or parents is None or any(not p.strip() for p in parents):
        raise ValueError("Expected nonempty trajectories with valid original parent IDs")
    if vocabulary != list(map(str, range(len(vocabulary)))):
        raise ValueError('Geography IDs must match vocabulary row positions')
    train, validation = create_gravity_samplers(dataset, config)
    train_indices, val_indices = split_trajectory_indices(
        parents, config.training.validation_split, config.training.shuffle_dataset,
        config.training.random_seed,
    )
    metadata = json.loads(gravity.with_suffix('.metadata.json').read_text())
    expected = {
        'dataset': config.data.dataset,
        'train_indices': train_indices, 'validation_indices': val_indices,
        'source_sha256': digest(source), 'gravity_sha256': digest(gravity),
        'validation_split': config.training.validation_split,
        'random_seed': config.training.random_seed,
        'shuffle': config.training.shuffle_dataset,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Gravity fit metadata mismatch: {key}; regenerate for this split")
    train_parents = sorted({parents[i] for i in train_indices})
    val_parents = sorted({parents[i] for i in val_indices})
    if set(train_parents) & set(val_parents):
        raise ValueError("Training and validation parents overlap")
    if not train_indices or not val_indices:
        raise ValueError('Training and deterministic validation must both be nonempty')
    weights = raw[train_indices]
    probabilities = train.weights / train.weights.sum()
    top_count = max(1, math.ceil(len(train_indices) * 0.01))
    diagnostics = {
        'dataset': config.data.dataset,
        'rows': len(dataset), 'train_rows': len(train_indices),
        'validation_rows': len(val_indices), 'train_parents': len(train_parents),
        'validation_parents': len(val_parents), 'parent_overlap': 0,
        'zero_weight_percent_all_rows': 100 * (raw == 0).double().mean().item(),
        'zero_weight_percent_training': 100 * (weights == 0).double().mean().item(),
        'total_raw_training_weight': weights.sum().item(),
        'min_raw_training_weight': weights.min().item(),
        'max_raw_training_weight': weights.max().item(),
        'top_one_percent_training_count': top_count,
        'top_one_percent_probability_mass': probabilities.topk(top_count).values.sum().item(),
        'effective_trajectory_count': 1 / probabilities.square().sum().item(),
        'source_sha256': digest(source), 'gravity_sha256': digest(gravity),
        'geography_sha256': digest(directory / 'roadmap.geo'),
        'relations_sha256': digest(directory / 'roadmap.rel'),
        'text_sha256': digest(root / f'Trajs_{config.data.dataset}.txt'),
        'statistics_scope': 'regional counts fitted on training parents only',
    }
    split = {**expected, 'train_parents': train_parents, 'validation_parents': val_parents}
    return base, dataset, relations, train, validation, diagnostics, split


@torch.no_grad()
def sample_and_check(model, base, relations, count, seed):
    """Record EOS versus the length cap explicitly using existing generation rules."""
    allowed = {tuple(map(int, pair)) for pair in relations}
    eos = base.stoi[base.EOS_TOKEN]
    samples = []
    model.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        for index in range(count):
            origin = base.stoi[base.origins[index % len(base.origins)]]
            sequence = torch.tensor([[eos, origin]], device='cuda')
            reason = 'length_cap'
            for _ in range(80):
                logits, _ = model(sequence[:, -model.block_size:])
                logits = logits[:, -1, :]
                # GPT.forward applies the shared adjacency membership mask,
                # just as it does for policy scoring and GPT.generate_test.
                probabilities = torch.softmax(logits, dim=-1)
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("Nonfinite generation probabilities")
                token = torch.multinomial(probabilities, 1)
                if token.item() == eos:
                    reason = 'eos'
                    break
                if sequence.shape[1] == 81:
                    break
                sequence = torch.cat((sequence, token), dim=1)
            roads = [int(base.itos[token]) for token in sequence[0, 1:].tolist()]
            connected = all(pair in allowed for pair in zip(roads, roads[1:]))
            valid_ids = all(str(road) in base.stoi for road in roads)
            samples.append({'roads': roads, 'termination': reason,
                            'valid_road_ids': valid_ids, 'connected': connected})
    return {
        'count': count, 'valid_road_ids': all(s['valid_road_ids'] for s in samples),
        'connected': all(s['connected'] for s in samples),
        'eos_count': sum(s['termination'] == 'eos' for s in samples),
        'length_cap_count': sum(s['termination'] == 'length_cap' for s in samples),
        'samples': samples,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='NYC20K')
    parser.add_argument('--data-root', type=Path, required=True, help='Prepared artifacts from this run')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--sample-count', type=int, default=0)
    parser.add_argument('--require-short-check', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--profile-only', action='store_true',
                        help='20 temporary in-memory updates plus validation; no trained checkpoints')
    args = parser.parse_args(argv)
    if (args.steps <= 0 or args.batch_size <= 0 or args.workers < 0
            or args.eval_every <= 0 or args.sample_count < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('Invalid training settings')
    if args.workers != 0:
        parser.error('Exact sampled-example exposure requires zero loader workers')
    return args


def build_config(args):
    config = get_base_config()
    config.data.dataset = args.dataset
    if getattr(args, 'data_root', None) is not None:
        config.data.root = str(args.data_root.resolve())
    config.data.block_size = config.data.max_length = 81
    config.model.use_lora = False
    # Keep LoRA tensors and names in the existing architecture; train all tensors.
    config.training.batch_size = args.batch_size
    config.training.num_workers = args.workers
    config.training.max_iters = 20 if args.profile_only else args.steps
    config.training.learning_rate = args.learning_rate
    config.training.device = config.system.device = 'cuda'
    config.model.merge_from_dict(config.model.model_configs[config.model.model_type])
    return config


def run(args):
    config = build_config(args)
    torch.set_num_threads(2)
    set_seed(config.system.seed)
    base, dataset, relations, train, validation, diagnostics, split = prepare_data(config)
    config.model.vocab_size = base.get_vocab_size()
    config.model.block_size = base.get_block_size()
    print(json.dumps(diagnostics, indent=2), flush=True)
    if args.validate_only:
        return
    if args.output is None:
        raise ValueError('--output is required for training or profiling')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    if not args.profile_only and args.steps > 50:
        if args.require_short_check is None:
            raise ValueError('Runs longer than 50 steps require --require-short-check')
        short = json.loads((args.require_short_check / 'results.json').read_text())
        previous = json.loads((args.require_short_check / 'config.json').read_text())
        check_short_compatibility(config, diagnostics, short, previous)
    args.output = require_output(args.output, [args.data_root])
    args.output.mkdir(parents=True, exist_ok=False)
    config.system.work_dir = str(args.output)
    write_json(args.output / 'split.json', split)
    write_json(args.output / 'gravity_diagnostics.json', diagnostics)
    write_json(args.output / 'run_arguments.json', vars(args) | {
        'data_root': str(args.data_root) if args.data_root else None,
        'output': str(args.output),
        'require_short_check': str(args.require_short_check) if args.require_short_check else None,
    })
    (args.output/'source_diff.patch').write_text(git_record('diff', '--binary'))
    snapshot_sources(args.output/'source_snapshots')
    write_json(args.output/'source_fingerprints.json', {name:digest(p) for name,p in source_entries().items()})
    torch.cuda.empty_cache()
    pre_free, total = torch.cuda.mem_get_info()
    budget = memory_budget(pre_free, total)
    write_json(args.output / 'memory_budget.json', {
        'pre_run_free_bytes': pre_free, 'gpu_total_bytes': total, 'budget_bytes': budget,
        'reserve_headroom_bytes': 512 * 1024**2, 'fraction_of_total_limit': 0.85,
        'externally_occupied_bytes_before_run': total - pre_free,
    })
    torch.cuda.reset_peak_memory_stats()
    run_started = time.monotonic()
    model, adjacency = initialize_model(config, base, relations)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    write_json(args.output / 'config.json', config.to_dict())
    parameter_count = sum(p.numel() for p in model.parameters())
    held_out = DeterministicTrajectoryWindows(dataset, validation.indices)
    exposure = ExposureDataset(dataset)
    trainer = Trainer(config.training, model, exposure, train, validation, val_dataset=held_out)
    watched = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
               if name in ('transformer.wte.weight', 'transformer.h.0.attn.c_attn.weight',
                           'transformer.h.0.mlp.c_fc.weight')}
    best = float('inf')
    gradient_checks = 0
    history = []

    def ensure_budget():
        if torch.cuda.max_memory_reserved() >= budget:
            raise MemoryBudgetExceeded('Peak reserved VRAM reached the pre-run memory budget')

    def validate(trainer):
        nonlocal best
        value = trainer.evaluate()
        ensure_budget()
        print(f'step {trainer.iter_num}: held-out token loss {value:.6f}', flush=True)
        history.append({'step': trainer.iter_num, 'validation_loss': value,
                        'validation_tokens': trainer.val_tokens})
        if value < best:
            best = value
            if not args.profile_only:
                torch.save(model.state_dict(), args.output / 'model_best.pt')

    def after_backward(trainer):
        nonlocal gradient_checks
        if any(p.grad is None for p in model.parameters()):
            raise RuntimeError('A full-model parameter has no gradient')
        # clip_grad_norm_(error_if_nonfinite=True) checks every gradient's norm.
        gradient_checks += 1
        ensure_budget()

    def batch_end(trainer):
        exposure.batch_ends.append((trainer.iter_num, len(exposure.draws)))
        row = {'step': trainer.iter_num, 'loss': trainer.loss.item(),
               'gradient_norm': trainer.grad_norm.item(), 'step_seconds': trainer.iter_dt}
        history.append(row)
        with (args.output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        if trainer.iter_num == 1 or trainer.iter_num % 10 == 0:
            print(json.dumps(row), flush=True)
        if trainer.iter_num % args.eval_every == 0 or trainer.iter_num == config.training.max_iters:
            validate(trainer)
            if not args.profile_only:
                torch.save(model.state_dict(), args.output / 'model_pretrain.pt')

    trainer.add_callback('on_after_backward', after_backward)
    trainer.add_callback('on_batch_end', batch_end)
    validate(trainer)
    initial_loss = trainer.val_loss
    trainer.run()
    changes = {name: (dict(model.named_parameters())[name].detach().cpu() - old).abs().max().item()
               for name, old in watched.items()}
    if not all(value > 0 and math.isfinite(value) for value in changes.values()):
        raise RuntimeError('Backbone update check failed')
    if args.profile_only:
        samples = {'count': 0, 'valid_road_ids': True, 'connected': True, 'eos_count': 0, 'length_cap_count': 0}
    else:
        checkpoint = args.output / 'model_pretrain.pt'
        model.eval()
        x, _ = held_out[0]
        x = x.unsqueeze(0).cuda()
        with torch.no_grad():
            expected_logits = model(x)[0]
            reloaded = GPT(config.model, adj_matrix=adjacency).cuda()
            reloaded.load_state_dict(torch.load(checkpoint, map_location='cuda', weights_only=True), strict=True)
            reloaded.eval()
            actual_logits = reloaded(x)[0]
            torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        samples = sample_and_check(reloaded, base, relations, args.sample_count, config.system.seed + 1)
    write_json(args.output / 'samples.json', samples)
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    ensure_budget()
    passed = (gradient_checks == config.training.max_iters and (args.profile_only or (
              samples['count'] > 0 and samples['valid_road_ids'] and samples['connected']
              and samples['eos_count'] > 0)))
    with (Path(config.data.root) / f'{config.data.dataset}-Taxi/{config.data.dataset}_Taxi_trajectory_train.csv').open(newline='') as handle:
        source_rows = list(csv.DictReader(handle))
    seen = Counter(i for i, _ in exposure.draws)
    with (args.output / 'sampled_examples.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['draw', 'step', 'trajectory_index', 'trajectory_id', 'parent_id', 'valid_prediction_targets'])
        start = 0
        for step, stop in exposure.batch_ends:
            for draw in range(start, stop):
                index, targets = exposure.draws[draw]
                row = source_rows[index]
                writer.writerow([draw, step, index, row['traj_id'], row['parent_traj_id'], targets])
            start = stop
        if start != len(exposure.draws) or len(exposure.batch_ends) != trainer.iter_num:
            raise RuntimeError('Consumed sample and training-step accounting differs')
    exposure_report = {
        'draws_consumed': len(exposure.draws), 'distinct_source_trajectories': len(seen),
        'distinct_original_parents': len({source_rows[i]['parent_traj_id'] for i in seen}),
        'distinct_roads_in_actual_windows': len(exposure.window_roads),
        'network_roads': base.get_vocab_size() - 1,
        'network_road_exposure_fraction': len(exposure.window_roads) / (base.get_vocab_size() - 1),
        'trajectory_draw_counts': {str(i): n for i, n in sorted(seen.items())},
        'actual_window_road_ids': sorted(exposure.window_roads), 'workers': 0,
    }
    write_json(args.output / 'sampled_exposure.json', exposure_report)
    results = {
        'passed': passed, 'steps': trainer.iter_num, 'finite_gradient_steps': gradient_checks,
        'dataset': config.data.dataset, 'profile_only': args.profile_only,
        'initialized_from_scratch': True, 'memory_budget_passed': True,
        'memory_budget_bytes': budget, 'pre_run_free_bytes': pre_free,
        'adjacency_dtype': str(adjacency.dtype), 'adjacency_storage_bytes': adjacency.numel() * adjacency.element_size(),
        'trainable_parameters': parameter_count, 'backbone_max_absolute_updates': changes,
        'checkpoint_reload': 'not performed; profiling saves no weights' if args.profile_only else 'strict state dict and exact logits passed',
        'initial_validation_loss': initial_loss, 'final_validation_loss': trainer.val_loss,
        'best_validation_loss': best, 'validation_windows': len(held_out),
        'validation_tokens': trainer.val_tokens,
        'evaluation': 'deterministic, unweighted, token-mean; every held-out target once',
        'peak_host_ram_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved(),
        'gpu_free_bytes_at_end': free, 'gpu_total_bytes': total,
        'elapsed_seconds': time.monotonic() - run_started,
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
        'sample_check': {key: value for key, value in samples.items() if key != 'samples'},
        'gravity_diagnostics': diagnostics,
        'sampled_exposure': {k: v for k, v in exposure_report.items()
                            if k not in ('trajectory_draw_counts', 'actual_window_road_ids')},
    }
    write_json(args.output / 'evaluation_history.json', [h for h in history if 'validation_loss' in h])
    write_json(args.output / 'results.json', results)
    print(json.dumps(results, indent=2), flush=True)
    if args.sample_count and not passed:
        raise RuntimeError('Run checks failed; inspect results.json before full training')


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (torch.cuda.OutOfMemoryError, MemoryBudgetExceeded) as exc:
        if args.output is not None and args.output.exists():
            failure = {'passed': False, 'profile_only': args.profile_only,
                       'failure': type(exc).__name__, 'error': str(exc),
                       'peak_vram_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_vram_reserved_bytes': torch.cuda.max_memory_reserved()}
            write_json(args.output / 'results.json', failure)
            print(json.dumps(failure), flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
