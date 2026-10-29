"""Fail before training on unavailable dependencies, invalid raw data or a bad network."""
import importlib
import math
from pathlib import Path

MISSING_NETWORK = ('Canonical assessment INCOMPLETE: no canonical network supplied; '
    'the run will compute 11 network-independent metrics and report off-network rate as unavailable. '
    'Prepared geometry is not a substitute for the canonical network.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_configuration(config):
    p, d, metrics = config['pretraining'], config['dpo'], config['metrics']
    for value in (p['updates'], p['safety_updates'], p['batch_size'], d['updates'],
                  d['preferences'], d['batch_size'], metrics['jobs']):
        require(isinstance(value, int) and value > 0, 'Update, batch, sample and worker counts must be positive integers')
    require(p['workers'] >= 0, 'Loader worker count cannot be negative')
    require(all(math.isfinite(x) and x > 0 for x in (p['learning_rate'], d['learning_rate'], d['beta'])),
            'Learning rates and beta must be positive and finite')
    require(d['selected_update'] in d['milestones'] and max(d['milestones']) <= d['updates'],
            'Selected update/milestones must be available within the DPO budget')
    require(p['gravity'] and p['adjacency'] and p['adjacency_dtype'] == 'torch.bool',
            'Established gravity/boolean-adjacency recipe required')
    require(config['evaluation']['prompt_roads'] == [1, 3] and config['evaluation']['seeds'] == [42, 43, 44]
            and config['evaluation']['all_eligible_retained_test_sections'], 'Evaluation coverage cannot be reduced')
    require(config['preprocessing']['candidate_road_limit'] > 0, 'Candidate-road limit must be positive')


def validate_network(settings):
    from . import metrics
    path = settings['canonical_network']
    if path is None:
        return dict(complete=False, unavailable_reason=MISSING_NETWORK)
    path = Path(path)
    require(path.is_file(), f'Supplied canonical network is not a file: {path}')
    _, _, record = metrics.load_road_spatial_index(path, settings['network_layer'], settings['network_mode_column'])
    return dict(complete=True, **record)


def validate_raw(input_dir):
    from . import preprocessing, metrics
    raw = Path(input_dir)
    roads, outgoing, incoming, relation_count = preprocessing.load_network(raw)
    geometry = metrics.load_geo_id_to_latlon(raw/'nyc.geo')
    require(set(geometry) == roads, 'Raw geometry/road identity mismatch')
    metrics.load_rel_adjacency(raw/'nyc.rel', geometry)
    train, parents = preprocessing.load_trips(raw/'nyc_mm_train.csv', roads, outgoing, incoming)
    test, test_parents = preprocessing.load_trips(raw/'nyc_mm_test.csv', roads, outgoing, incoming, parents)
    return dict(roads=len(roads), relations=relation_count, training_parents=len(train),
                test_parents=len(test), parent_overlap=len(parents & test_parents))


def validate_cuda():
    import torch
    require(torch.cuda.is_available(), 'CUDA is required; CPU fallback is disabled. Check GPU/device access.')
    x = torch.ones((128, 128), device='cuda')
    require((x @ x)[0, 0].item() == 128, 'CUDA matrix multiplication failed')
    torch.cuda.synchronize()
    return dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                matrix_multiply='passed', free_and_total_bytes=list(torch.cuda.mem_get_info()))


def validate(config):
    validate_configuration(config)
    from . import metrics
    versions = metrics.dependencies(config['metrics']['canonical_network'] is not None)
    for name in ('torch', 'pandas', 'geopandas', 'shapely', 'geopy', 'opacus', 'transformers',
                 'nyc_workflow.pretraining', 'nyc_workflow.dpo', 'preprocessing.gravity_based_sampling'):
        importlib.import_module(name)
    # Invalid supplied networks fail even when the no-network policy would permit
    # incomplete scores. This happens before preprocessing, safety or training.
    network = validate_network(config['metrics'])
    raw = validate_raw(config['input_dir'])
    cuda = validate_cuda()
    return dict(passed=True, dependencies=versions, raw_inputs=raw, cuda=cuda, canonical_network=network)
