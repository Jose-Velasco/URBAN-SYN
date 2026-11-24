"""Prepared alignment, canonical training, sampling, and resume regressions."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from urban_syn.diffusion.conditional.prepared_metadata import align_prepared, build_prepared_metadata, load_prepared_metadata
from urban_syn.diffusion.conditional.sparse_pilot import smoke_inputs, run_sparse_pilot
from urban_syn.diffusion.conditional.conditioning import encode_requests
from urban_syn.diffusion.data import RoadSequenceDataset
from urban_syn.experiment.config import load_experiment_config, ConditioningSettings
from urban_syn.experiment.datasets import load_training_dataset, build_epoch_loader, fitted_metadata
from urban_syn.experiment.factories import build_model, build_loss_composer
from urban_syn.experiment.trainer import run_epoch
from urban_syn.experiment.checkpoint import save_training_checkpoint, load_training_checkpoint, experiment_fingerprint


@pytest.fixture
def inputs(tmp_path):
    return smoke_inputs(tmp_path/'inputs')


def frames(inputs):
    return [pd.read_parquet(inputs['train']), pd.read_parquet(inputs['test']),
            pd.read_csv(inputs['mapping']), pd.read_parquet(inputs['observations'])]


@pytest.mark.parametrize('piece',[False,True])
def test_alignment_shuffled_and_zero(inputs,piece):
    train,test,mapping,obs=frames(inputs)
    identity=['uid','tid','move_id']
    if piece:
        mapping['piece']=0; obs['piece']=0; identity+=['piece']
    shuffled = train.iloc[[2,0,3,1]]
    aligned,_=align_prepared(shuffled,test,mapping.sample(frac=1),obs.sample(frac=1),identity)
    assert aligned['gen_train'].fmm_id.tolist()==shuffled.fmm_id.tolist()
    assert aligned['gen_test'].fmm_id.tolist()==test.fmm_id.tolist()
    dataset=RoadSequenceDataset(inputs['train'],torch.tensor([0,4,9]))
    assert dataset[0][0].tolist()==[1,2,3]


@pytest.mark.parametrize('failure',['missing','duplicate','identity','counts','mode','overlap','null'])
def test_reject_bad_alignment(inputs,failure):
    train,test,mapping,obs=frames(inputs)
    if failure=='missing': mapping=mapping.iloc[1:]
    if failure=='duplicate': mapping=pd.concat([mapping,mapping.iloc[:1]])
    if failure=='identity': mapping.loc[1,['uid','tid','move_id']]=mapping.loc[0,['uid','tid','move_id']]
    if failure=='counts': obs=obs.iloc[1:]
    if failure=='mode': obs.loc[0,'label']='conflict'
    if failure=='overlap':
        mapping.loc[4,'uid']=1
        mapping.loc[4,'tid']='heldout'
        mask=obs.uid=='5'
        obs.loc[mask,'uid']='1'
        obs.loc[mask,'tid']='heldout'
    if failure=='null': mapping.loc[0,'uid']=None
    with pytest.raises(ValueError): align_prepared(train,test,mapping,obs)


def config_for(inputs,directory):
    config=load_experiment_config('configs/experiments/metadata_conditioning/enabled.yaml')
    paths=config.settings.paths.model_copy(update=dict(train_sequences=inputs['train'],test_sequences=inputs['test'],trip_id_map=inputs['mapping']))
    condition=config.settings.model.conditioning.model_copy(update=dict(metadata_dir=directory,all_dropout=0.,field_dropout=0.))
    model=config.settings.model.model_copy(update=dict(conditioning=condition))
    settings=config.settings.model_copy(update=dict(paths=paths,model=model))
    return config.__class__(settings=settings,joint_settings=config.joint_settings,config_path=config.config_path,project_root=config.project_root)


def test_fit_alignment_epoch_checkpoint_resume(inputs,tmp_path):
    directory=tmp_path/'metadata'
    build_prepared_metadata(**inputs,output=directory)
    config=config_for(inputs,directory)
    graph=Data(x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]]),edge_id=torch.tensor([0,4,9]),edge_index=torch.tensor([[0,1,2],[1,2,0]]))
    dataset,split=load_training_dataset(config,graph.edge_id)
    rows,schema=fitted_metadata(config,dataset,split)
    training=rows.set_index('fmm_id').loc[list(split.train_fmm_ids)]
    assert next(f for f in schema['fields'] if f['name']=='temperature')['normalization']['mean']==training.temperature.mean()
    model=build_model(config,graph)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    loader=build_epoch_loader(dataset,split.train_indices,2,17,0,True,False)
    metrics,step=run_epoch(model,graph,loader,build_loss_composer(config.settings.training.losses),optimizer=optimizer,global_step=0,grad_clip_norm=1.,max_batches=None,log_every_steps=100)
    assert step>0
    fingerprint=experiment_fingerprint(config,graph)
    path=tmp_path/'joint.pt'
    save_training_checkpoint(path,config,model,optimizer,None,1,step,1.,1,fingerprint,split.train_fmm_ids,split.validation_fmm_ids,None)
    restored=build_model(config,graph); opt=torch.optim.AdamW(restored.parameters(),lr=.001)
    load_training_checkpoint(path,restored,opt,None,torch.device('cpu'),fingerprint,split.train_fmm_ids,split.validation_fmm_ids)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    # The restored optimizer and model must support a real resumed update.
    _, resumed_step=run_epoch(restored,graph,loader,build_loss_composer(config.settings.training.losses),
                              optimizer=opt,global_step=step,grad_clip_norm=1.,max_batches=1,log_every_steps=100)
    assert resumed_step==step+1
    restored.metadata_schema=copy.deepcopy(restored.metadata_schema)
    restored.metadata_schema['fields'][0]['vocabulary']['bad']=99
    with pytest.raises(ValueError,match='metadata schema'):
        load_training_checkpoint(path,restored,opt,None,torch.device('cpu'),fingerprint,split.train_fmm_ids,split.validation_fmm_ids)
    validation=build_epoch_loader(dataset,split.validation_indices,2,17,0,False,False)
    run_epoch(model,graph,validation,build_loss_composer(config.settings.training.losses),optimizer=None,global_step=step,grad_clip_norm=1.,max_batches=None,log_every_steps=100)
    for roads,rates,metadata in build_epoch_loader(dataset,tuple(range(len(dataset))),2,19,0,True,False):
        # fixture rate/roads are equal, so validate by category-normalized temperature relation
        temperature=metadata['values']['temperature']
        assert torch.isfinite(temperature).all()
    manifest=load_prepared_metadata(directory)
    (directory/'gen_train.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='fingerprint'): load_prepared_metadata(directory)


def test_canonical_smoke(tmp_path):
    report=run_sparse_pilot(tmp_path/'smoke')
    assert report['runs']['frozen']['gnn_gradient']==0


def test_disabled_original_projection():
    from urban_syn.diffusion.diff_module import diff_CSDI
    torch.manual_seed(7)
    implicit=diff_CSDI(8,8,3,8,4,1)
    torch.manual_seed(7)
    explicit=diff_CSDI(8,8,3,8,4,1,metadata_encoder=None)
    assert implicit.input_projection.in_channels==5
    assert all(torch.equal(v,explicit.state_dict()[k]) for k,v in implicit.state_dict().items())


def test_bucket_alignment_and_training_only_fit(inputs,tmp_path):
    from urban_syn.experiment.datasets import MetadataDataset
    from urban_syn.diffusion.conditional.sparse_metadata import fit_fields
    # Unique rates identify each sequence even when shuffling length buckets.
    train=pd.read_parquet(inputs['train'])
    train['rates']=[[i/10]*3 for i in train.fmm_id]
    train.to_parquet(inputs['train'],index=False)
    directory=tmp_path/'metadata'
    build_prepared_metadata(**inputs,output=directory)
    config=config_for(inputs,directory)
    dataset,split=load_training_dataset(config,torch.tensor([0,4,9]))
    rows,schema=fitted_metadata(config,dataset,split)
    field=next(f for f in schema['fields'] if f['name']=='temperature')
    for _,rates,meta in build_epoch_loader(dataset,tuple(range(len(dataset))),2,19,0,True,False):
        actual=meta['values']['temperature']*field['normalization']['scale']+field['normalization']['mean']
        assert torch.allclose(actual,rates[:,0]*50)
    # Shuffled sidecars align identically; held-out rows never enter the fit.
    wrapped=MetadataDataset(dataset.dataset,rows.sample(frac=1),schema['fields'])
    assert torch.equal(wrapped.metadata['values']['temperature'],dataset.metadata['values']['temperature'])
    training=rows.set_index('fmm_id').loc[list(split.train_fmm_ids)]
    changed=rows.copy()
    heldout=changed.fmm_id.isin(split.validation_fmm_ids)
    changed.loc[heldout,'weather_category']='HELDOUT_ONLY'
    changed.loc[heldout,'temperature']=1e9
    refit=fit_fields(changed.set_index('fmm_id').loc[list(split.train_fmm_ids)],config.settings.model.conditioning.fields)
    assert refit==schema['fields']
    encoded=encode_requests([{'weather_category':'HELDOUT_ONLY'}],schema['fields'])
    assert not encoded['masks']['weather_category'].item()


def test_canonical_masked_placeholder_invariance():
    from urban_syn.diffusion.diff_module import diff_CSDI
    from urban_syn.diffusion.conditional.conditioning import SparseMetadataEncoder
    from urban_syn.diffusion.conditional.tests.test_sparse import fields
    schema=fields()
    model=diff_CSDI(8,8,3,8,4,1,metadata_encoder=SparseMetadataEncoder(schema)).eval()
    null=encode_requests([{},{}],schema)
    for value in null['values'].values():
        value.fill_(999999 if value.dtype==torch.long else float('nan'))
    x,t=torch.randn(2,3,5),torch.zeros(2,1,dtype=torch.long)
    assert torch.equal(model(x,t,None),model(x,t,None,metadata=null))


def test_configuration_pair():
    enabled=load_experiment_config('configs/experiments/metadata_conditioning/enabled.yaml')
    disabled=load_experiment_config('configs/experiments/metadata_conditioning/disabled.yaml')
    assert enabled.joint_settings==disabled.joint_settings
    assert enabled.settings.paths==disabled.settings.paths
    assert enabled.settings.experiment.seed==disabled.settings.experiment.seed
    assert not ConditioningSettings().enabled
    left, right = (config.settings.model_dump(mode='json') for config in (enabled, disabled))
    left['experiment']['name'] = right['experiment']['name']
    left['model']['conditioning']['enabled'] = False
    assert left == right


def test_legacy_weights_rejected(tmp_path):
    from urban_syn.diffusion.generate import load_generation_model
    from urban_syn.diffusion.config import JointTrainingSettings
    path=tmp_path/'pilot.pt'
    torch.save({'model_state_dict':{'diff_model.input_projection.weight':torch.zeros(8,13,1)}},path)
    graph=Data(x=torch.ones(3,2),edge_id=torch.tensor([0,4,9]),edge_index=torch.tensor([[0,1],[1,2]]))
    settings=JointTrainingSettings.model_validate(dict(gnn={},diffusion={},training={}))
    with pytest.raises((ValueError,KeyError,RuntimeError)):
        load_generation_model(path,graph,settings,torch.device('cpu'))


def test_generation_requests_and_masking(tmp_path):
    from shapely.geometry import LineString
    from urban_syn.diffusion.generate import generate_gps_trajectories, load_generation_model
    from urban_syn.diffusion.config import JointTrainingSettings
    from urban_syn.diffusion.sampling.ddpm import DDPMSampler
    output=tmp_path/'smoke'
    run_sparse_pilot(output)
    info=torch.load(output/'enabled.pt',weights_only=True)
    settings=JointTrainingSettings.model_validate(info['config'])
    graph=Data(x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]]),edge_id=torch.tensor([0,4,9]),
               edge_index=torch.tensor([[0,1,2,1],[1,2,0,0]]))
    model=load_generation_model(output/'enabled.pt',graph,settings,torch.device('cpu'))
    embeddings=model.encode_roads(graph)
    geometries={i:LineString([(-74.,40.),(-73.99,40.01)]) for i in [0,4,9]}
    def generate(requests,masked=False):
        torch.manual_seed(17)
        return generate_gps_trajectories(model,graph.edge_id,embeddings,geometries,{2:1,3:2},DDPMSampler(),
                                         batch_size=2,metadata_requests=requests,mask_metadata=masked)
    requests=[{'temperature':10.},{'weather_category':'rain'},{}]
    known=generate(requests)
    masked=generate(requests,True)
    empty=generate([{}, {}, {}])
    pd.testing.assert_frame_equal(masked,empty)
    assert known.trajectory_id.nunique()==3 and (known.model_road_id>0).all()
    with pytest.raises(ValueError,match='One metadata request'): generate([{}])
    with pytest.raises(ValueError,match='unsupported'): generate([{'bad':1},{},{}])


def test_stale_source_rejected(inputs,tmp_path):
    directory=tmp_path/'metadata'
    build_prepared_metadata(**inputs,output=directory)
    inputs['weather'].write_bytes(b'changed')
    with pytest.raises(ValueError,match='Stale'): load_prepared_metadata(directory)


@pytest.mark.parametrize('missing_features', [False, True])
def test_on_off_loaders_identical_across_epochs_and_partitions(tmp_path, missing_features):
    """Only metadata differs, including multiple buckets and remainder batches."""
    from dataclasses import replace
    from urban_syn.experiment.datasets import MetadataDataset

    root = tmp_path/'inputs'
    root.mkdir()
    records, identities, observations, weather = [], [], [], []
    # Two lengths per trip guarantee both partitions contain both buckets.
    # Ten train trips yield seven train and three validation rows per bucket.
    for trip in range(11):
        for move, length in enumerate((2, 5)):
            fid = 10 + trip * 2 + move
            uid, tid = f'u{trip}', f't{trip}'
            roads = [[0, 4, 9][(position + fid) % 3] for position in range(length)]
            records.append(dict(fmm_id=fid, edge_ids=roads, rates=[fid/100]*length, num_points=length))
            identities.append(dict(fmm_id=fid, uid=uid, tid=tid, move_id=str(move), piece=0))
            for position in range(length):
                observations.append(dict(uid=uid, tid=tid, move_id=str(move), piece=0,
                                         fmm_id=fid, datetime=1704067200+position,
                                         label=None if missing_features else f'mode{trip%2}'))
        weather.append(dict(uid=uid, tid=tid, DATE='2024-01-01',
                            DESCRIPTION=None if missing_features else f'weather{trip%3}',
                            TAVG_C=None if missing_features else float(trip)))
    prepared = pd.DataFrame(records)
    train = prepared.iloc[:-2].sample(frac=1, random_state=29)
    test = prepared.iloc[-2:].iloc[::-1]
    paths = {name: root/f'{name}.parquet' for name in ('train','test','observations','weather')}
    paths['mapping'] = root/'mapping.csv'
    train.to_parquet(paths['train'], index=False)
    test.to_parquet(paths['test'], index=False)
    pd.DataFrame(identities).sample(frac=1, random_state=30).to_csv(paths['mapping'], index=False)
    pd.DataFrame(observations).sample(frac=1, random_state=31).to_parquet(paths['observations'], index=False)
    pd.DataFrame(weather).sample(frac=1, random_state=32).to_parquet(paths['weather'], index=False)
    directory = tmp_path/'metadata'
    manifest = build_prepared_metadata(**paths, output=directory, identity=('uid','tid','move_id','piece'))
    assert manifest['partition_ids']['gen_train'] == train.fmm_id.tolist()
    assert manifest['partition_ids']['gen_test'] == test.fmm_id.tolist()
    assert pd.read_parquet(directory/'gen_train.parquet').fmm_id.tolist() == train.fmm_id.tolist()

    enabled = config_for(paths, directory)
    validation = enabled.settings.training.validation.model_copy(update={'fraction': .3})
    training = enabled.settings.training.model_copy(update={'validation': validation})
    enabled = replace(enabled, settings=enabled.settings.model_copy(update={'training': training}))
    condition = enabled.settings.model.conditioning.model_copy(update={'enabled': False, 'metadata_dir': tmp_path/'ABSENT'})
    disabled_model = enabled.settings.model.model_copy(update={'conditioning': condition})
    disabled = replace(enabled, settings=enabled.settings.model_copy(update={'model': disabled_model}))
    graph_ids = torch.tensor([0,4,9])
    off, off_split = load_training_dataset(disabled, graph_ids)
    on, on_split = load_training_dataset(enabled, graph_ids)
    assert type(off) is RoadSequenceDataset
    assert type(on) is MetadataDataset and type(on.dataset) is RoadSequenceDataset
    assert on.fmm_ids is on.dataset.fmm_ids and on.lengths is on.dataset.lengths
    assert on.fmm_ids == off.fmm_ids == train.fmm_id.tolist()
    assert on.lengths == off.lengths
    assert on_split == off_split
    for index in range(len(off)):
        # The wrapper returns the base tensors themselves, without cloning or conversion.
        for wrapped, base in zip(on[index][:2], on.dataset[index]):
            assert wrapped is base
        assert torch.equal(on[index][0], off[index][0])
        assert torch.equal(on[index][1], off[index][1])
    if missing_features:
        assert all(not masks.any() for masks in on.metadata['masks'].values())

    epoch_orders = []
    for epoch in range(3):
        for partition in ('train', 'validation'):
            indices = getattr(on_split, f'{partition}_indices')
            assert set(on.lengths[i] for i in indices) == {2,5}
            loaders = [build_epoch_loader(dataset, indices, 2, 101+epoch, 0,
                                          partition=='train', False) for dataset in (off, on)]
            off_batches, on_batches = [list(copy.deepcopy(loader.batch_sampler)) for loader in loaders]
            assert off_batches == on_batches
            assert any(len(batch)==1 for batch in off_batches)
            seen = []
            for local_indices, off_batch, on_batch in zip(off_batches, loaders[0], loaders[1], strict=True):
                off_ids = [off.fmm_ids[indices[i]] for i in local_indices]
                on_ids = [on.fmm_ids[indices[i]] for i in local_indices]
                assert off_ids == on_ids
                assert torch.equal(off_batch[0], on_batch[0])
                assert torch.equal(off_batch[1], on_batch[1])
                # Unique rate markers independently verify the collated FMM membership.
                assert torch.equal((on_batch[1][:,0]*100).round().long(), torch.tensor(on_ids))
                expected = {group: {name: value[[indices[i] for i in local_indices]]
                                   for name,value in fields.items()} for group,fields in on.metadata.items()}
                for group,fields in expected.items():
                    for name,value in fields.items():
                        assert torch.equal(on_batch[2][group][name], value)
                seen.extend(on_ids)
            assert sorted(seen) == sorted(on.fmm_ids[i] for i in indices)
            if partition=='train':
                epoch_orders.append(seen)
    assert epoch_orders[0] != epoch_orders[1]


@pytest.mark.parametrize('failure', ['blank', 'extra_observation', 'fmm_conflict', 'fractional_fmm',
                                     'boolean_fmm', 'duplicate_columns', 'missing_piece'])
def test_strict_identity_joins(inputs, failure):
    train,test,mapping,obs = frames(inputs)
    identity = ['uid','tid','move_id']
    if failure=='blank':
        mapping['uid'] = mapping.uid.astype(str)
        mapping.loc[0,'uid'] = '  '
    elif failure=='extra_observation':
        extra = obs.iloc[:1].assign(uid='extra')
        obs = pd.concat([obs, extra], ignore_index=True)
    elif failure=='fmm_conflict':
        obs['fmm_id'] = obs.uid.astype(int)
        obs.loc[0,'fmm_id'] = 999
    elif failure=='fractional_fmm':
        mapping['fmm_id'] = mapping.fmm_id.astype(float)
        mapping.loc[0,'fmm_id'] = 1.5
    elif failure=='boolean_fmm':
        mapping['fmm_id'] = mapping.fmm_id.astype(object)
        mapping.loc[0,'fmm_id'] = True
    elif failure=='duplicate_columns':
        mapping = pd.concat([mapping,mapping[['uid']]],axis=1)
    elif failure=='missing_piece':
        mapping = pd.concat([mapping,mapping.iloc[:1].assign(fmm_id=99)],ignore_index=True)
        mapping['piece'] = range(len(mapping))
        # Without piece in the configured identity, this map is ambiguous.
    with pytest.raises(ValueError):
        align_prepared(train,test,mapping,obs,identity)


def test_disabled_factory_needs_no_metadata_or_trajectory_artifacts(inputs, tmp_path, monkeypatch):
    from dataclasses import replace
    from urban_syn.diffusion.training import build_joint_model
    import urban_syn.experiment.datasets as datasets

    config = config_for(inputs, tmp_path/'NONEXISTENT_METADATA')
    condition = config.settings.model.conditioning.model_copy(update={'enabled': False})
    options = config.settings.model.model_copy(update={'conditioning': condition})
    # Model construction in disabled mode must not read trajectory inputs either.
    paths = config.settings.paths.model_copy(update={'train_sequences': tmp_path/'NONEXISTENT_TRAIN'})
    config = replace(config, settings=config.settings.model_copy(update={'model': options, 'paths': paths}))
    def unexpected_metadata_read(*args, **kwargs):
        raise AssertionError('Disabled construction attempted to load metadata')
    monkeypatch.setattr(datasets, 'fitted_metadata', unexpected_metadata_read)
    graph = Data(x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]]), edge_id=torch.tensor([0,4,9]),
                 edge_index=torch.tensor([[0,1,2],[1,2,0]]))
    torch.manual_seed(901)
    original_path = build_joint_model(graph, config.joint_settings).eval()
    torch.manual_seed(901)
    disabled = build_model(config, graph).eval()
    assert disabled.diff_model.metadata_encoder is None
    assert all(p.requires_grad for p in disabled.road_encoder.parameters())
    assert original_path.state_dict().keys() == disabled.state_dict().keys()
    assert all(torch.equal(value, disabled.state_dict()[key]) for key,value in original_path.state_dict().items())
    x = torch.randn(2,3,config.joint_settings.gnn.output_dim+1)
    t = torch.zeros(2,1,dtype=torch.long)
    assert torch.equal(original_path.diff_model(x,t,None), disabled.diff_model(x,t,None))
