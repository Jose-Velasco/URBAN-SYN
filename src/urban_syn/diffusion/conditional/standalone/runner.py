"""Standalone check/train/generate/evaluate, with no GNN model dependency."""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from urban_syn.experiment.datasets import MetadataDataset, build_epoch_loader, split_prepared_sequences
from ..conditioning import encode_requests, slice_metadata
from ..metadata import sha256
from ..prepared_metadata import load_prepared_metadata, _sequence_order
from ..sparse_metadata import fit_fields
from .core import FixedRoads, FixedDiffusion, digest

DEFAULTS = dict(
    conditioning=dict(enabled=False, fields=['weather_category','transportation_mode','temperature']),
    model=dict(channels=64, steps=100, residual_layers=4, condition_dim=8, all_dropout=.1,
               field_dropout=.1, beta_start=.0001,beta_end=.02,comparison_chunk=256),
    training=dict(epochs=1,batch_size=8,learning_rate=.001,weight_decay=.01,validation_fraction=.2,
                  max_batches=None,grad_clip=1.),
    seeds=dict(initialization=42,metadata=314,batch_order=101,split=101,training_noise=202,evaluation_noise=303,generation=404),
    device='cpu',output='outputs/standalone',
)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    allowed = set(DEFAULTS)|{'paths'}
    if set(raw)-allowed:
        raise ValueError(f'Unknown configuration keys: {set(raw)-allowed}')
    config = copy.deepcopy(DEFAULTS)
    for name,value in raw.items():
        if name in DEFAULTS and isinstance(DEFAULTS[name],dict):
            if set(value)-set(DEFAULTS[name]):
                raise ValueError(f'Unknown {name} settings')
            config[name].update(value)
        else:
            config[name]=value
    required = {'train','test','identity_map','embeddings','network'}
    if not required.issubset(config.get('paths',{})):
        raise ValueError(f'Required paths: {sorted(required)}')
    if set(config['paths'])-required-{'metadata','geometry','geometry_layer'}:
        raise ValueError('Unknown source path settings')
    max_batches=config['training']['max_batches']
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError('training.max_batches must be null or a positive integer')
    if not 0 < config['training']['validation_fraction'] < 1:
        raise ValueError('A nonempty train-only validation partition is required')
    for group, names in [('training',['epochs','batch_size','learning_rate','grad_clip']),
                         ('model',['channels','steps','residual_layers','condition_dim','comparison_chunk'])]:
        if any(config[group][name]<=0 for name in names):
            raise ValueError(f'Positive {group} budgets required')
    m=config['model']
    if m['channels']<4 or m['channels']%2 or not 0<m['beta_start']<=m['beta_end']<1:
        raise ValueError('Invalid diffusion width or noise schedule')
    if not all(0<=m[key]<=1 for key in ['all_dropout','field_dropout']):
        raise ValueError('Dropout must be in [0,1]')
    return config


class Context:
    def __init__(self, config):
        self.config=config
        p=config['paths']
        self.output=Path(config['output'])
        self.output.mkdir(parents=True,exist_ok=True)
        self.device=torch.device(config['device'])
        self.roads=FixedRoads(p['embeddings'],p['network'],[p['train'],p['test']],self.output/'missing_roads.json')
        self.train=self.roads.dataset(p['train'])
        self.test=self.roads.dataset(p['test'])
        self.split=split_prepared_sequences(self.train.fmm_ids,Path(p['identity_map']),
                                           config['training']['validation_fraction'],config['seeds']['split'],True)
        mapping=pd.read_csv(p['identity_map'],dtype={'uid':str,'tid':str})
        if mapping.fmm_id.duplicated().any() or mapping[['fmm_id','uid','tid']].isna().any().any():
            raise ValueError('Incomplete or duplicate identity map')
        indexed=mapping.set_index('fmm_id')
        train_users=set(indexed.loc[self.train.fmm_ids].uid)
        test_users=set(indexed.loc[self.test.fmm_ids].uid)
        if train_users & test_users:
            raise ValueError('Train/test users overlap')
        self.fields=[]; self.rows={}; self.provenance=None; self.temperature_bins=[]
        if config['conditioning']['enabled']:
            directory=Path(p['metadata'])
            self.provenance=load_prepared_metadata(directory)
            for key,source in [('train','train'),('test','test'),('identity_map','mapping')]:
                if sha256(p[key])!=self.provenance['sources'][source]['sha256']:
                    raise ValueError(f'Metadata source mismatch: {key}')
            for name,dataset in [('train',self.train),('test',self.test)]:
                self.rows[name]=_sequence_order(pd.read_parquet(directory/f'gen_{name}.parquet'),dataset.fmm_ids)
            fitting=self.rows['train'].iloc[list(self.split.train_indices)]
            self.fields=fit_fields(fitting,config['conditioning']['fields'])
            if 'temperature' in fitting:
                values=fitting.temperature.dropna().to_numpy(dtype=float)
                self.temperature_bins=sorted(set(np.quantile(values,[0,.25,.5,.75,1]).tolist())) if len(values) else []
            self.train=MetadataDataset(self.train,self.rows['train'],self.fields)
            self.test=MetadataDataset(self.test,self.rows['test'],self.fields)
        self.schema=dict(version=1,fields=self.fields,temperature_bins=self.temperature_bins,provenance=self.provenance)
        self.fingerprints=dict(**self.roads.fingerprints, **{key:sha256(p[key]) for key in ['train','test','identity_map']})
        self.partition_ids=dict(train=list(self.split.train_fmm_ids),validation=list(self.split.validation_fmm_ids),test=self.test.fmm_ids)
        compatibility=copy.deepcopy(config)
        for key in ['output','device']: compatibility.pop(key)
        compatibility['training'].pop('epochs')
        self.identity=digest(dict(config=compatibility,schema=self.schema,inputs=self.fingerprints,partitions=self.partition_ids))

    def model(self):
        c=self.config
        return FixedDiffusion(c['model'],self.roads,self.fields,c['conditioning']['enabled'],
                              c['seeds']['initialization'],c['seeds']['metadata']).to(self.device)

    def loader(self, split, epoch=0):
        dataset=self.test if split=='test' else self.train
        indices=tuple(range(len(dataset))) if split=='test' else getattr(self.split,split+'_indices')
        return build_epoch_loader(dataset,indices,self.config['training']['batch_size'],
                                  self.config['seeds']['batch_order']+epoch,0,split=='train',False)

    def metadata(self,split):
        if not self.fields: return None
        dataset=self.test if split=='test' else self.train
        indices=list(range(len(dataset))) if split=='test' else list(self.split.validation_indices)
        return slice_metadata(dataset.metadata,indices)


def move(metadata,device):
    return None if metadata is None else {g:{k:v.to(device) for k,v in fields.items()} for g,fields in metadata.items()}


def altered(metadata, mode, seed):
    if metadata is None: return None
    result=copy.deepcopy(metadata)
    if mode=='masked':
        for mask in result['masks'].values(): mask.zero_()
    elif mode=='shuffled':
        count=len(next(iter(result['values'].values())))
        permutation=torch.randperm(count,generator=torch.Generator().manual_seed(seed))
        # A tiny held-out set can randomly receive the identity permutation.
        # Ensure a real row-order perturbation whenever more than one row exists.
        if count>1 and torch.equal(permutation,torch.arange(count)):
            permutation=permutation.roll(1)
        result=slice_metadata(result,permutation)
    return result


def save(ctx,model,optimizer,epoch,best,path):
    payload=dict(format='standalone-fixed-v1',identity=ctx.identity,config=ctx.config,
                 schema=ctx.schema,fingerprints=ctx.fingerprints,partitions=ctx.partition_ids,
                 model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best=best,
                 rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                          python=random.getstate(),numpy=[np.random.get_state()[0],np.random.get_state()[1].tolist(),
                                                         *np.random.get_state()[2:]]))
    path=Path(path); temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary); temporary.replace(path)


def load(ctx,model,path,optimizer=None):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('format')!='standalone-fixed-v1':
        raise ValueError('Legacy/joint checkpoints cannot resume in the standalone runner')
    if checkpoint['identity']!=ctx.identity or checkpoint['schema']!=ctx.schema or checkpoint['fingerprints']!=ctx.fingerprints:
        raise ValueError('Standalone checkpoint inputs/settings/schema mismatch')
    model.load_state_dict(checkpoint['model'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng']['torch'])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])
        random.setstate(checkpoint['rng']['python'])
        numpy_state=checkpoint['rng']['numpy']
        np.random.set_state((numpy_state[0],np.asarray(numpy_state[1],dtype=np.uint32),*numpy_state[2:]))
    return checkpoint


@torch.no_grad()
def denoising(ctx,model,split,mode='correct'):
    model.eval()
    # Dataset indices recover metadata after bucketing; noise is reset per variant.
    dataset=ctx.test if split=='test' else ctx.train
    indices=tuple(range(len(dataset))) if split=='test' else ctx.split.validation_indices
    all_meta=altered(ctx.metadata(split),mode,ctx.config['seeds']['evaluation_noise']+1)
    loader=ctx.loader(split)
    # Copy sampler so inspecting indices never advances the live epoch sampler.
    batches=list(copy.deepcopy(loader.batch_sampler))
    generator=torch.Generator(device=ctx.device).manual_seed(ctx.config['seeds']['evaluation_noise'])
    records=[]
    for local,batch in zip(batches,loader,strict=True):
        meta=move(slice_metadata(all_meta,local),ctx.device) if all_meta is not None else None
        noise,road=model.losses(batch[0].to(ctx.device),batch[1].to(ctx.device),meta,generator)
        for offset,n,r in zip(local,noise.tolist(),road.tolist()):
            records.append(dict(fmm_id=dataset.fmm_ids[indices[offset]],noise_mse=n,reconstruction_mse=r))
    return records


def train(ctx,resume=None):
    if (ctx.output/'last.pt').exists() and resume is None:
        raise FileExistsError('Standalone run exists; choose a fresh output or use --resume')
    torch.manual_seed(ctx.config['seeds']['initialization'])
    model=ctx.model(); options=ctx.config['training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    start=0; best=float('inf')
    if resume:
        saved=load(ctx,model,resume,optimizer); start=saved['epoch']; best=saved['best']
    original=model.fixed_embeddings.clone()
    history=[]
    for epoch in range(start,options['epochs']):
        model.train()
        generator=torch.Generator(device=ctx.device).manual_seed(ctx.config['seeds']['training_noise']+epoch)
        losses=[]
        for step,batch in enumerate(ctx.loader('train',epoch)):
            if options['max_batches'] is not None and step>=options['max_batches']: break
            optimizer.zero_grad(set_to_none=True)
            meta=move(batch[2],ctx.device) if len(batch)==3 else None
            noise,road=model.losses(batch[0].to(ctx.device),batch[1].to(ctx.device),meta,generator)
            loss=(noise+road).mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite standalone objective')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),options['grad_clip'])
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite standalone gradients')
            optimizer.step(); losses.append(loss.item())
        validation=denoising(ctx,model,'validation')
        score=float(np.mean([r['noise_mse']+r['reconstruction_mse'] for r in validation]))
        if not np.isfinite(score): raise FloatingPointError('Nonfinite validation loss')
        improved=score<best; best=min(best,score)
        assert torch.equal(original,model.fixed_embeddings)
        save(ctx,model,optimizer,epoch+1,best,ctx.output/'last.pt')
        if improved: save(ctx,model,optimizer,epoch+1,best,ctx.output/'best.pt')
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=score)
        history.append(record)
        with (ctx.output/'history.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
    (ctx.output/'manifest.json').write_text(json.dumps(dict(schema=ctx.schema,inputs=ctx.fingerprints,
                                                        partitions=ctx.partition_ids,config=ctx.config),indent=2)+'\n')
    return history


@torch.no_grad()
def generate(ctx,model,split='test',mode='correct'):
    model.eval()
    dataset=ctx.test if split=='test' else ctx.train
    indices=list(range(len(dataset))) if split=='test' else list(ctx.split.validation_indices)
    metadata=altered(ctx.metadata(split),mode,ctx.config['seeds']['evaluation_noise']+1)
    generator=torch.Generator(device=ctx.device).manual_seed(ctx.config['seeds']['generation'])
    records=[]
    # One request per held-out trajectory preserves its length and metadata composition.
    for offset,index in enumerate(indices):
        meta=move(slice_metadata(metadata,[offset]),ctx.device) if metadata is not None else None
        roads,rates=model.generate(1,dataset.lengths[index],meta,generator)
        records.append(dict(fmm_id=dataset.fmm_ids[index],edge_ids=roads[0].tolist(),rates=rates[0].tolist(),num_points=dataset.lengths[index]))
    return pd.DataFrame(records)


def geometry(ctx):
    path=ctx.config['paths'].get('geometry')
    if path is None: return None
    if sha256(path)!=ctx.roads.fingerprints['network']:
        raise ValueError('GPS geometry must match the fixed embedding network hash')
    from ...reconstruction import load_road_geometries
    result=load_road_geometries(path,layer=ctx.config['paths'].get('geometry_layer','roads'))
    if not set(ctx.roads.edge_ids.tolist()).issubset(result): raise ValueError('Incomplete matching road geometry')
    return result


def gps(frame,geometries):
    from ...reconstruction import reconstruct_gps_trajectory
    records=[]
    for row in frame.itertuples():
        points=reconstruct_gps_trajectory(row.edge_ids,row.rates,geometries)
        for step,(edge,rate,(lon,lat,position)) in enumerate(zip(row.edge_ids,row.rates,points)):
            records.append(dict(fmm_id=row.fmm_id,step=step,edge_id=edge,rate=rate,lng=lon,lat=lat,position_rate=position))
    return pd.DataFrame(records)


def distribution_metrics(ctx, generated, reference, geometries=None):
    from collections import Counter
    def counts(frame): return Counter(int(road) for seq in frame.edge_ids for road in seq)
    actual,expected=counts(generated),counts(reference)
    vocab=ctx.roads.edge_ids.tolist()
    p=np.array([actual[i] for i in vocab],dtype=float); p/=max(p.sum(),1)
    q=np.array([expected[i] for i in vocab],dtype=float); q/=max(q.sum(),1)
    midpoint=(p+q)/2
    def kl(x,y):
        mask=x>0
        return float((x[mask]*np.log(x[mask]/y[mask])).sum())
    pairs=[(int(a),int(b)) for seq in generated.edge_ids for a,b in zip(seq[:-1],seq[1:])]
    unique=len({tuple(seq) for seq in generated.edge_ids})
    result=dict(count=len(generated),transition_count=len(pairs),
                # All adjacent observation pairs count, including repeated-road pairs.
                # Staying on one road is valid without an explicit graph self-loop.
                invalid_transition_fraction=sum(a != b and (a,b) not in ctx.roads.pairs for a,b in pairs)/len(pairs) if pairs else None,
                road_visitation_js_divergence=(kl(p,midpoint)+kl(q,midpoint))/2,
                unique_trajectory_fraction=unique/len(generated) if len(generated) else None,
                road_entropy=float(-(p[p>0]*np.log(p[p>0])).sum()),
                trip_distance=None,trip_distance_status='matching geometry not supplied')
    if geometries is not None:
        from ...reconstruction import _haversine_meters, reconstruct_gps_trajectory
        def distances(frame):
            values=[]
            for row in frame.itertuples():
                pts=reconstruct_gps_trajectory(row.edge_ids,row.rates,geometries)
                values.append(sum(_haversine_meters(a[:2],b[:2]) for a,b in zip(pts[:-1],pts[1:])))
            return np.array(values)
        a,b=distances(generated),distances(reference)
        quantiles=np.linspace(0,1,max(len(a),len(b)))
        result['trip_distance']=dict(generated_mean_m=float(a.mean()),reference_mean_m=float(b.mean()),
                                     quantile_wasserstein_m=float(np.abs(np.quantile(a,quantiles)-np.quantile(b,quantiles)).mean()))
        result['trip_distance_status']='GPS observation-to-observation haversine path length; not complete routed distance'
    return result


def evaluate(ctx, model, split, reporting=None):
    reporting=reporting or ctx
    dataset=ctx.test if split=='test' else ctx.train
    indices=list(range(len(dataset))) if split=='test' else list(ctx.split.validation_indices)
    frame=pd.read_parquet(ctx.config['paths']['test' if split=='test' else 'train']).set_index('fmm_id')
    ids=[dataset.fmm_ids[i] for i in indices]
    reference=frame.loc[ids].reset_index()
    geometries=geometry(ctx)
    report=dict(split=split,checkpoint_selection='validation loss only',
                seeds=ctx.config['seeds'],schema=reporting.schema,inputs=ctx.fingerprints,
                composition='one generated trajectory per held-out FMM ID, same length and metadata request',variants={})
    groups={}
    if reporting.fields:
        rows=reporting.rows['test' if split=='test' else 'train'].set_index('fmm_id').loc[ids]
        encoded=reporting.metadata(split)
        for field in reporting.fields:
            name=field['name']; labels=[]
            for value,known in zip(rows[name],encoded['masks'][name].tolist()):
                if not known: labels.append('<missing-or-training-unseen>')
                elif field['kind']=='categorical': labels.append(str(value))
                else:
                    cuts=reporting.temperature_bins[1:-1]
                    labels.append(f'train_temperature_bin_{np.searchsorted(cuts,float(value),side="right")}')
            groups[name]={label:[fid for fid,l in zip(ids,labels) if l==label] for label in sorted(set(labels))}
        report['missingness']={name:dict(total=len(ids),unavailable=int((~mask).sum()))
                               for name,mask in encoded['masks'].items()}
    else:
        report['conditional_status']='No reporting metadata supplied; pass --report-metadata for off-run category comparisons'
    for mode in ['correct','masked','shuffled']:
        losses=pd.DataFrame(denoising(reporting,model,split,mode)).set_index('fmm_id')
        generated=generate(reporting,model,split,mode)
        def summarize(membership):
            selected=losses.loc[membership]
            metrics=distribution_metrics(ctx,generated.set_index('fmm_id').loc[membership].reset_index(),
                                         reference.set_index('fmm_id').loc[membership].reset_index(),geometries)
            return dict(noise_mse=float(selected.noise_mse.mean()),reconstruction_mse=float(selected.reconstruction_mse.mean()),**metrics)
        report['variants'][mode]=dict(pooled=summarize(ids),
                                      by_field={name:{category:summarize(members) for category,members in categories.items()}
                                                for name,categories in groups.items()})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','train','generate','evaluate'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--device',help='cpu, cuda, or cuda:N (after CUDA_VISIBLE_DEVICES remapping)')
    parser.add_argument('--checkpoint')
    parser.add_argument('--resume')
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--metadata-mode',choices=['correct','masked','shuffled'],default='correct')
    parser.add_argument('--report-metadata',help='Prepared metadata directory for stratifying a disabled checkpoint')
    args=parser.parse_args(argv)
    config=load_config(args.config)
    if args.device: config['device']=args.device
    required=['train','test','identity_map','embeddings','network']
    if config['conditioning']['enabled']: required.append('metadata')
    missing={name:config['paths'].get(name) for name in required
             if not config['paths'].get(name) or not Path(config['paths'][name]).exists()}
    if missing:
        result=dict(status='missing_inputs',missing=missing,
                    matching_observations='Required for metadata preparation; exact sampled/unsampled source is not available without its provenance manifest',
                    legacy_nyc_embeddings='Unverified: new network identity and complete canonical road coverage are required')
        Path(config['output']).mkdir(parents=True,exist_ok=True)
        (Path(config['output'])/'missing_inputs.json').write_text(json.dumps(result,indent=2)+'\n')
        if args.command!='check': raise FileNotFoundError(json.dumps(result))
        print(json.dumps(result,indent=2))
        return result
    ctx=Context(config)
    if args.command=='check':
        result=dict(status='passed',inputs=ctx.fingerprints,partitions=ctx.partition_ids,
                    conditioning=config['conditioning']['enabled'],fixed_embedding_shape=list(ctx.roads.embeddings.shape))
    elif args.command=='train':
        result=train(ctx,args.resume)
    else:
        model=ctx.model()
        if not args.checkpoint: parser.error('--checkpoint is required for generation/evaluation')
        load(ctx,model,args.checkpoint)
        reporting=ctx
        if args.report_metadata:
            config=copy.deepcopy(config)
            config['conditioning']['enabled']=True
            config['paths']['metadata']=args.report_metadata
            reporting=Context(config)
        if args.command=='generate':
            frame=generate(reporting,model,args.split,args.metadata_mode)
            path=ctx.output/f'generated_{args.split}_{args.metadata_mode}.parquet'
            frame.to_parquet(path,index=False)
            geometries=geometry(ctx)
            if geometries is not None: gps(frame,geometries).to_parquet(path.with_name(path.stem+'_gps.parquet'),index=False)
            result=dict(output=str(path),count=len(frame))
        else:
            result=evaluate(ctx,model,args.split,reporting)
            result['checkpoint']=str(Path(args.checkpoint).resolve())
            result['checkpoint_sha256']=sha256(args.checkpoint)
            (ctx.output/f'evaluation_{args.split}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result
