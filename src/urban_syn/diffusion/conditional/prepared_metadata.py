"""Strict alignment and enrichment for canonical prepared sequence Parquets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .metadata import sha256
from .sparse_metadata import (
    DEFAULT_FIELDS, POI_FIELD, _timestamp_seconds, _prepare_poi_assignments,
    _validate_parent_disjointness, prepare_rows,
)

VERSION = 3


def _identity_columns(frame, identity, source):
    if not frame.columns.is_unique or not set(identity).issubset(frame):
        raise ValueError(f"{source}: missing or ambiguous identity columns")
    if frame[identity].isna().any().any():
        raise ValueError(f"{source}: missing complete trajectory identity")
    for column in identity:
        if not frame[column].map(pd.api.types.is_scalar).all():
            raise ValueError(f"{source}: identity values must be scalar")
        frame[column] = frame[column].astype(str)
        if frame[column].str.strip().eq('').any():
            raise ValueError(f"{source}: empty trajectory identity")


def _fmm_ids(frame, source):
    if not frame.columns.is_unique or 'fmm_id' not in frame:
        raise ValueError(f"{source}: missing or ambiguous FMM ID column")
    raw = frame.fmm_id
    ids = pd.to_numeric(raw, errors='raise')
    if (raw.map(lambda value: isinstance(value, bool)).any() or ids.isna().any()
            or (ids % 1 != 0).any()):
        raise ValueError(f"{source}: FMM IDs must be complete integers")
    try:
        frame['fmm_id'] = ids.astype('int64')
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{source}: FMM IDs must fit signed int64") from exc


def _sequence_order(rows, fmm_ids):
    """Require exact coverage and explicitly restore upstream sequence order."""
    if rows.fmm_id.duplicated().any() or set(rows.fmm_id) != set(fmm_ids):
        raise ValueError("Incomplete or ambiguous sequence identity coverage")
    return rows.set_index('fmm_id').loc[list(fmm_ids)].reset_index()


def align_prepared(train, test, mapping, observations, identity=("uid", "tid", "move_id"),
                   timestamp="datetime", mode="label"):
    """Align exact FMM observations; preserve sequence order and missing features.

    Identity columns must distinguish every map record (including pieces when
    present). Extra observation identities are rejected rather than aggregated
    into a partial join. Repeated observation rows remain observations: their
    counts must agree with the prepared sequence, and no points are deduplicated.
    """
    identity = list(identity)
    if (not {'uid', 'tid'}.issubset(identity) or len(identity) != len(set(identity))
            or set(identity) & {'fmm_id', timestamp, mode}):
        raise ValueError("Configure distinct trajectory identity columns including uid and tid")
    train, test, mapping, observations = (frame.copy() for frame in (train, test, mapping, observations))
    for source, frame in [('mapping', mapping), ('observations', observations)]:
        _identity_columns(frame, identity, source)
    for source, frame in [('mapping', mapping), ('train', train), ('test', test)]:
        _fmm_ids(frame, source)
        if frame.fmm_id.duplicated().any():
            raise ValueError(f"{source}: duplicate FMM IDs")
    if mapping.duplicated(identity).any():
        raise ValueError("Duplicate trajectory identities; configure the full piece-level identity")
    # Bidirectional identity coverage prevents a sampled/unsampled subset from
    # being silently accepted when only some observations happen to match.
    coverage = mapping[identity].merge(observations[identity].drop_duplicates(),
                                       on=identity, how='outer', indicator=True, validate='one_to_one')
    if not coverage['_merge'].eq('both').all():
        raise ValueError("Incomplete mapping/observation identity coverage")
    if 'fmm_id' in observations:
        _fmm_ids(observations, 'observations')
        association = observations[['fmm_id', *identity]].merge(
            mapping[['fmm_id', *identity]], on=identity, suffixes=('_observed', '_mapped'),
            validate='many_to_one')
        if not association.fmm_id_observed.eq(association.fmm_id_mapped).all():
            raise ValueError("Observation FMM ID conflicts with mapped identity")
    if timestamp not in observations:
        raise ValueError("Observations lack departure/end timestamp source")
    observations['_seconds'] = _timestamp_seconds(observations[timestamp], timestamp)
    grouped = observations.groupby(identity, sort=False)
    if mode in observations and (grouped[mode].nunique(dropna=True) > 1).any():
        raise ValueError("Conflicting mode labels within a trajectory")
    summary = grouped['_seconds'].agg(start_timestamp='min', end_timestamp='max', observation_count='size').reset_index()
    if mode in observations:
        summary['label'] = grouped[mode].first().to_numpy()
    aligned_map = mapping[['fmm_id', *identity]].merge(summary, on=identity, how='left', validate='one_to_one', sort=False)
    aligned = {}
    for split, sequences in [('gen_train', train), ('gen_test', test)]:
        required = {'fmm_id', 'edge_ids', 'rates', 'num_points'}
        if not required.issubset(sequences):
            raise ValueError("Prepared sequences lack required trajectory columns")
        ambiguous = (set(sequences) & set(aligned_map)) - {'fmm_id'}
        if ambiguous:
            raise ValueError(f"Ambiguous prepared metadata columns: {sorted(ambiguous)}")
        rows = sequences.merge(aligned_map, on='fmm_id', how='left', validate='one_to_one', sort=False)
        rows = _sequence_order(rows, sequences.fmm_id.tolist())
        if rows[identity + ['observation_count']].isna().any().any():
            raise ValueError("Incomplete sequence identity coverage")
        if any(len(r.edge_ids) != r.num_points or len(r.rates) != r.num_points or r.observation_count != r.num_points for r in rows.itertuples()):
            raise ValueError("Prepared/source observation count mismatch")
        rows['date'] = pd.to_datetime(rows.start_timestamp, unit='s', utc=True).dt.strftime('%Y-%m-%d')
        aligned[split] = rows.drop(columns=['edge_ids', 'rates'])
    if set(train.fmm_id) & set(test.fmm_id):
        raise ValueError("Train/test FMM IDs overlap")
    _validate_parent_disjointness(aligned)
    return aligned, aligned_map


def build_prepared_metadata(train, test, mapping, observations, weather, output,
                            identity=("uid", "tid", "move_id"), timestamp='datetime', mode='label',
                            poi=None, stops=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new metadata destination: {output}")
    if (poi is None) != (stops is None):
        raise ValueError("POI and stops must be supplied together")
    sources = dict(train=train, test=test, mapping=mapping, observations=observations, weather=weather)
    if poi is not None:
        sources.update(poi=poi, stops=stops)
    aligned, complete = align_prepared(pd.read_parquet(train), pd.read_parquet(test),
                                      pd.read_csv(mapping, dtype={name: str for name in identity}), pd.read_parquet(observations), identity, timestamp, mode)
    assignments = None
    if poi is not None:
        assignments = _prepare_poi_assignments(aligned, complete, pd.read_parquet(poi), pd.read_parquet(stops))
    weather_rows = pd.read_parquet(weather)
    enriched = {split: _sequence_order(prepare_rows(rows, weather_rows, assignments), rows.fmm_id.tolist())
                for split, rows in aligned.items()}
    manifest = dict(version=VERSION, sources={k: dict(path=str(Path(v).resolve()), sha256=sha256(v)) for k,v in sources.items()},
                    alignment=dict(identity=list(identity), timestamp=timestamp, mode=mode, key='fmm_id',
                                   count_policy='exact source observations', split_policy='upstream; user-disjoint',
                                   order_policy='exact input sequence fmm_id order',
                                   identity_coverage='bidirectional mapping/observations'),
                    fields=list(DEFAULT_FIELDS) + ([POI_FIELD] if poi else []),
                    partition_ids={k: rows.fmm_id.tolist() for k, rows in enriched.items()}, artifacts={})
    output.mkdir(parents=True)
    for split, rows in enriched.items():
        path = output / f'{split}.parquet'
        rows.to_parquet(path, index=False)
        manifest['artifacts'][path.name] = sha256(path)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def load_prepared_metadata(directory):
    """Reject stale inputs and modified sidecars before fitting or use."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('version') != VERSION:
        raise ValueError("Unsupported prepared metadata manifest")
    for source in manifest['sources'].values():
        if sha256(source['path']) != source['sha256']:
            raise ValueError("Stale metadata source")
    for name, digest in manifest['artifacts'].items():
        if sha256(directory/name) != digest:
            raise ValueError("Metadata artifact fingerprint mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'test', 'mapping', 'observations', 'weather', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--identity', nargs='+', default=['uid', 'tid', 'move_id'])
    parser.add_argument('--timestamp', default='datetime')
    parser.add_argument('--mode', default='label')
    parser.add_argument('--poi', type=Path)
    parser.add_argument('--stops', type=Path)
    build_prepared_metadata(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
