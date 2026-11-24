"""Package a verified word2vec table and explicit vocabulary; never train embeddings.

Provenance must originate from the embedding producer and bind all four inputs.
An old table without verified network provenance cannot be made compatible by
assigning the new network's hash to it.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..metadata import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['embedding-text','vocabulary','transitions','network','provenance','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--vocabulary-index-base',type=int,choices=[0,1],default=1)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    provenance=json.loads(args.provenance.read_text())
    for key,path in [('embedding_text',args.embedding_text),('vocabulary',args.vocabulary),
                     ('transitions',args.transitions),('network',args.network)]:
        if provenance.get(key+'_sha256')!=sha256(path):
            raise ValueError(f'Unverified {key}: matching producer provenance is required')
    with args.embedding_text.open() as stream:
        count,width=map(int,stream.readline().split())
        table=np.loadtxt(stream,ndmin=2)
    if table.shape!=(count,width+1) or not np.isfinite(table).all():
        raise ValueError('Invalid embedding text shape/values')
    if not np.array_equal(table[:,0],np.arange(count)):
        raise ValueError('Text row IDs must be explicit zero-based row indices')
    def unique_object(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('Duplicate vocabulary keys')
            result[key]=value
        return result
    vocabulary=json.loads(args.vocabulary.read_text(),object_pairs_hook=unique_object)
    keys={str(i+args.vocabulary_index_base) for i in range(count)}
    if set(vocabulary)!=keys: raise ValueError('Vocabulary does not exactly cover embedding rows')
    ids=np.array([vocabulary[str(i+args.vocabulary_index_base)] for i in range(count)])
    if ids.dtype.kind not in 'iu' or (ids<0).any() or len(set(ids.tolist()))!=count:
        raise ValueError('Vocabulary values must be unique nonnegative raw road integers')
    pairs=pd.read_csv(args.transitions)[['source','target']].to_numpy()
    if pairs.dtype.kind not in 'iu' or not set(pairs.ravel()).issubset(set(ids.tolist())):
        raise ValueError('Transitions must contain raw road IDs in this exact vocabulary')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez(stream,edge_ids=ids,embeddings=table[:,1:].astype(np.float32),transitions=pairs,
                 network_sha256=sha256(args.network))


if __name__=='__main__': main()
