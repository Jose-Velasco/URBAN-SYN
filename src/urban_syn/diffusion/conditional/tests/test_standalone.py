"""Synthetic end-to-end coverage of the fixed-embedding standalone runner."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from urban_syn.diffusion.conditional.sparse_pilot import smoke_inputs
from urban_syn.diffusion.conditional.prepared_metadata import build_prepared_metadata
from urban_syn.diffusion.conditional.metadata import sha256
from urban_syn.diffusion.conditional.standalone.core import FixedRoads
from urban_syn.diffusion.conditional.standalone.runner import (
    Context, load_config, train, load, generate, evaluate, main,
)


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(2)
    paths=smoke_inputs(tmp_path/'inputs')
    # A matching GeoPackage permits real geometry export in the synthetic smoke.
    import geopandas as gpd
    from shapely.geometry import LineString
    network=tmp_path/'network.gpkg'
    gpd.GeoDataFrame(dict(edge_id=[0,4,9]),geometry=[LineString([(-74+i*.001,40),(-74+i*.001,40.01)]) for i in range(3)],crs=4326).to_file(network,layer='roads')
    bundle=tmp_path/'roads.npz'
    np.savez(bundle,edge_ids=np.array([9,0,4]),embeddings=np.eye(3,dtype=np.float32),
             transitions=np.array([[0,4],[4,9],[9,0]]),network_sha256=sha256(network))
    # Multiple held-out trajectories make shuffling a nontrivial diagnostic.
    test=pd.read_parquet(paths['test'])
    test=pd.concat([test,test.assign(fmm_id=6),test.assign(fmm_id=7)],ignore_index=True)
    test.to_parquet(paths['test'],index=False)
    mapping=pd.read_csv(paths['mapping'])
    mapping=pd.concat([mapping,mapping.iloc[[-1]].assign(fmm_id=6,uid=6),mapping.iloc[[-1]].assign(fmm_id=7,uid=7)],ignore_index=True)
    mapping.to_csv(paths['mapping'],index=False)
    observations=pd.read_parquet(paths['observations'])
    template=observations[observations.uid=='5']
    observations=pd.concat([observations,template.assign(uid='6',label='bus'),template.assign(uid='7')],ignore_index=True)
    observations.to_parquet(paths['observations'],index=False)
    weather=pd.read_parquet(paths['weather'])
    template=weather[weather.uid=='5']
    weather=pd.concat([weather,template.assign(uid='6',DESCRIPTION='rain',TAVG_C=10.),template.assign(uid='7',TAVG_C=15.)],ignore_index=True)
    weather.to_parquet(paths['weather'],index=False)
    directory=tmp_path/'metadata'
    build_prepared_metadata(**paths,output=directory)
    config=dict(paths=dict(train=str(paths['train']),test=str(paths['test']),identity_map=str(paths['mapping']),
                           embeddings=str(bundle),network=str(network),metadata=str(directory),geometry=str(network)),
                conditioning=dict(enabled=True),model=dict(channels=8,steps=3,residual_layers=1,comparison_chunk=2),
                training=dict(epochs=1,batch_size=2),output=str(tmp_path/'enabled'))
    path=tmp_path/'enabled.yaml'; path.write_text(yaml.safe_dump(config))
    enabled=load_config(path)
    disabled=copy.deepcopy(enabled); disabled['conditioning']['enabled']=False
    disabled['output']=str(tmp_path/'disabled'); disabled['paths']['metadata']=str(tmp_path/'ABSENT')
    return path,enabled,disabled


def test_parity_immutability_resume_and_generation(setup):
    path,enabled,disabled=setup
    a,b=Context(enabled),Context(disabled)
    assert a.split==b.split and a.train.fmm_ids==b.train.fmm_ids
    for epoch in [0,1]:
        x,y=list(a.loader('train',epoch)),list(b.loader('train',epoch))
        assert len(x)==len(y)
        for first,second in zip(x,y):
            assert torch.equal(first[0],second[0]) and torch.equal(first[1],second[1])
    before=torch.get_rng_state().clone()
    on,off=a.model(),b.model()
    assert torch.equal(before,torch.get_rng_state())
    width=off.net.input_projection.in_channels
    for name,value in off.net.state_dict().items():
        if name=='input_projection.weight':
            assert torch.equal(value,on.net.state_dict()[name][:,:width])
        else: assert torch.equal(value,on.net.state_dict()[name])
    fixed=on.fixed_embeddings.clone()
    assert not on.fixed_embeddings.requires_grad
    train(a); train(b)
    trained=a.model(); saved=load(a,trained,a.output/'last.pt')
    assert torch.equal(fixed,trained.fixed_embeddings)
    output=generate(a,trained)
    assert output.fmm_id.tolist()==a.test.fmm_ids
    assert set(road for seq in output.edge_ids for road in seq).issubset({0,4,9})
    assert all(len(ids)==count for ids,count in zip(output.edge_ids,output.num_points))
    again=a.model(); load(a,again,a.output/'last.pt')
    pd.testing.assert_frame_equal(output,generate(a,again))
    extended=copy.deepcopy(enabled); extended['training']['epochs']=2
    resumed=Context(extended); train(resumed,a.output/'last.pt')
    info=torch.load(a.output/'last.pt',weights_only=True)
    assert info['epoch']==2 and info['optimizer']
    uninterrupted=copy.deepcopy(enabled)
    uninterrupted['output']=str(Path(enabled['output']).parent/'uninterrupted')
    uninterrupted['training']['epochs']=2
    continuous=Context(uninterrupted); train(continuous)
    uninterrupted_info=torch.load(continuous.output/'last.pt',weights_only=True)
    assert all(torch.equal(v,uninterrupted_info['model'][key]) for key,v in info['model'].items())
    report=evaluate(a,trained,'test')
    assert set(report['variants'])=={'correct','masked','shuffled'}
    assert report['variants']['correct']['pooled']['trip_distance'] is not None
    assert report['missingness']['temperature']['total']==len(a.test)
    assert not any('road_encoder' in name for name,_ in trained.named_modules())
    main(['generate','--config',str(path),'--checkpoint',str(a.output/'last.pt')])
    assert (a.output/'generated_test_correct_gps.parquet').exists()
    # Disabled evaluation can use the same held-out composition for reporting.
    disabled_model=b.model(); load(b,disabled_model,b.output/'last.pt')
    off_report=evaluate(b,disabled_model,'test',a)
    assert off_report['variants']['correct']==off_report['variants']['masked']==off_report['variants']['shuffled']


@pytest.mark.parametrize('problem',['network','duplicate_mapping','nonfinite','missing_road','nonpositive_fmm','zero_length'])
def test_invalid_inputs_fail_without_filtering(setup,problem,tmp_path):
    _,config,_=setup
    p=config['paths']
    if problem=='zero_length':
        frame=pd.read_parquet(p['train']); frame.loc[0,'num_points']=0; frame.to_parquet(p['train'],index=False)
    elif problem=='nonpositive_fmm':
        frame=pd.read_parquet(p['train']); frame.loc[0,'fmm_id']=0; frame.to_parquet(p['train'],index=False)
    else:
        with np.load(p['embeddings']) as data: bundle={name:data[name].copy() for name in data.files}
        if problem=='network': bundle['network_sha256']=np.array('wrong')
        if problem=='duplicate_mapping': bundle['edge_ids']=np.array([0,0,4])
        if problem=='nonfinite': bundle['embeddings'][0,0]=float('nan')
        if problem=='missing_road':
            bundle['edge_ids']=np.array([9,0,99]); bundle['transitions']=np.array([[0,9]])
        np.savez(p['embeddings'],**bundle)
    with pytest.raises(ValueError): Context(config)
    if problem=='missing_road':
        assert (Path(config['output'])/'missing_roads.json').exists()
        assert len(pd.read_parquet(p['train']))==4


def test_all_missing_and_mismatched_checkpoint(setup):
    _,config,_=setup
    rows=pd.read_parquet(Path(config['paths']['metadata'])/'gen_train.parquet')
    # Existing metadata encoder contract: fully absent values remain trajectory rows.
    ctx=Context(config)
    for fields in ctx.train.metadata['masks'].values(): fields.zero_()
    model=ctx.model()
    batch=next(iter(ctx.loader('train')))
    losses=model.losses(batch[0],batch[1],batch[2])
    assert all(torch.isfinite(value).all() for value in losses)
    train(ctx)
    changed=copy.deepcopy(config); changed['seeds']['split']+=1
    with pytest.raises(ValueError,match='mismatch'): load(Context(changed),Context(changed).model(),ctx.output/'last.pt')
    legacy=ctx.output/'legacy.pt'; torch.save({'model_state_dict':model.state_dict()},legacy)
    with pytest.raises(ValueError,match='Legacy'): load(ctx,model,legacy)


def test_cli_check_and_config_pairs(setup):
    path,_,_=setup
    assert main(['check','--config',str(path)])['status']=='passed'
    root=Path('configs/experiments/metadata_conditioning/standalone')
    for budget in ['smoke','full']:
        on=load_config(root/f'{budget}_enabled.yaml'); off=load_config(root/f'{budget}_disabled.yaml')
        on['conditioning']['enabled']=False; on['output']=off['output']
        assert on==off


def test_verified_text_packaging(setup,tmp_path,monkeypatch):
    from urban_syn.diffusion.conditional.standalone.pack_embeddings import main as pack
    _,config,_=setup
    network=Path(config['paths']['network'])
    text=tmp_path/'embed.txt'; text.write_text('3 2\n0 1 0\n1 0 1\n2 1 1\n')
    vocabulary=tmp_path/'vocabulary.json'; vocabulary.write_text('{"1":9,"2":0,"3":4}')
    transitions=tmp_path/'transitions.csv'; transitions.write_text('source,target\n0,4\n4,9\n9,0\n')
    provenance=tmp_path/'producer.json'
    import json
    provenance.write_text(json.dumps({key+'_sha256':sha256(path) for key,path in
                                     [('embedding_text',text),('vocabulary',vocabulary),('transitions',transitions),('network',network)]}))
    output=tmp_path/'packaged.npz'
    args=['pack','--embedding-text',str(text),'--vocabulary',str(vocabulary),'--transitions',str(transitions),
          '--network',str(network),'--provenance',str(provenance),'--output',str(output)]
    monkeypatch.setattr('sys.argv',args)
    pack()
    roads=FixedRoads(output,network,[config['paths']['train'],config['paths']['test']],tmp_path/'missing.json')
    assert roads.edge_ids.tolist()==[9,0,4]
    bad_output=tmp_path/'unverified.npz'; args[-1]=str(bad_output)
    provenance.write_text('{}')
    monkeypatch.setattr('sys.argv',args)
    with pytest.raises(ValueError,match='Unverified'): pack()
    assert not bad_output.exists()


def test_diagnostics_share_noise_and_shuffle_complete_rows(setup,monkeypatch):
    from urban_syn.diffusion.conditional.standalone.runner import denoising, altered
    _,config,_=setup
    ctx=Context(config); model=ctx.model()
    original=model.losses
    seen=[]
    def capture(ids,rates,metadata=None,generator=None):
        seen.append(generator.get_state().clone())
        return original(ids,rates,metadata,generator)
    monkeypatch.setattr(model,'losses',capture)
    states=[]
    for mode in ['correct','masked','shuffled']:
        seen.clear(); denoising(ctx,model,'test',mode); states.append(list(seen))
    for a,b,c in zip(*states,strict=True):
        assert torch.equal(a,b) and torch.equal(b,c)
    metadata=ctx.metadata('test')
    shuffled=altered(metadata,'shuffled',304)
    assert not torch.equal(metadata['values']['temperature'],shuffled['values']['temperature'])
    for name in metadata['masks']:
        assert metadata['masks'][name].sum()==shuffled['masks'][name].sum()


@pytest.mark.parametrize('value', ['2', 'null', 'MAX_4ES', 'MAX_64ES', True, False, 0, -1, 2.0, [], {}])
def test_reject_invalid_max_batches(setup, value, tmp_path):
    path,_,_=setup
    raw=yaml.safe_load(path.read_text())
    raw['training']['max_batches']=value
    candidate=tmp_path/'invalid_budget.yaml'
    candidate.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError,match=r'training.max_batches'):
        load_config(candidate)


@pytest.mark.parametrize('value', [None, 1, 2])
def test_accept_valid_max_batches(setup, value, tmp_path):
    path,_,_=setup
    raw=yaml.safe_load(path.read_text())
    raw['training']['max_batches']=value
    candidate=tmp_path/'valid_budget.yaml'
    candidate.write_text(yaml.safe_dump(raw))
    assert load_config(candidate)['training']['max_batches'] == value


@pytest.mark.parametrize('name', ['smoke_enabled', 'smoke_disabled', 'full_enabled', 'full_disabled'])
def test_shipped_settings_perform_actual_updates(setup, name, tmp_path, monkeypatch):
    """Run the shipped architecture and batch settings on enough synthetic rows.

    Only input/output paths, CPU device, and full-run epoch count are adapted.
    Smoke must stop after two updates; full must consume all three batches.
    """
    _,fixture,_=setup
    p=fixture['paths']
    train_rows=pd.read_parquet(p['train'])
    mapping=pd.read_csv(p['identity_map'],dtype={'uid':str,'tid':str,'move_id':str})
    metadata_manifest=json.loads((Path(p['metadata'])/'manifest.json').read_text())
    obs_path=metadata_manifest['sources']['observations']['path']
    weather_path=metadata_manifest['sources']['weather']['path']
    observations=pd.read_parquet(obs_path)
    weather=pd.read_parquet(weather_path)
    extra_ids=range(100,246)  # 150 training trajectories total, 135 after validation.
    obs_template=observations[observations.uid=='1']
    weather_template=weather[weather.uid=='1']
    train_rows=pd.concat([train_rows,*[train_rows.iloc[[0]].assign(fmm_id=i) for i in extra_ids]],ignore_index=True)
    mapping=pd.concat([mapping,*[mapping[mapping.fmm_id==1].assign(fmm_id=i,uid=str(i)) for i in extra_ids]],ignore_index=True)
    observations=pd.concat([observations,*[obs_template.assign(uid=str(i)) for i in extra_ids]],ignore_index=True)
    weather=pd.concat([weather,*[weather_template.assign(uid=str(i)) for i in extra_ids]],ignore_index=True)
    train_rows.to_parquet(p['train'],index=False)
    mapping.to_csv(p['identity_map'],index=False)
    observations.to_parquet(obs_path,index=False)
    weather.to_parquet(weather_path,index=False)
    new_metadata=tmp_path/'expanded_metadata'
    build_prepared_metadata(train=p['train'],test=p['test'],mapping=p['identity_map'],
                            observations=obs_path,weather=weather_path,output=new_metadata)
    shipped=Path('configs/experiments/metadata_conditioning/standalone')/f'{name}.yaml'
    raw=yaml.safe_load(shipped.read_text())
    assert raw['training']['max_batches'] == (2 if name.startswith('smoke') else None)
    raw['paths']=dict(p,metadata=str(new_metadata))
    raw['output']=str(tmp_path/name)
    raw['device']='cpu'
    raw['training']['epochs']=1
    candidate=tmp_path/f'{name}.yaml'
    candidate.write_text(yaml.safe_dump(raw))
    config=load_config(candidate)
    ctx=Context(config)
    model=ctx.model()
    before=model.net.input_projection.weight.detach().clone()
    fixed=model.fixed_embeddings.clone()
    monkeypatch.setattr(ctx,'model',lambda: model)
    history=train(ctx)
    checkpoint=torch.load(ctx.output/'last.pt',weights_only=True)
    updates={int(state['step']) for state in checkpoint['optimizer']['state'].values() if 'step' in state}
    assert updates == ({2} if name.startswith('smoke') else {3})
    assert np.isfinite(history[0]['train_loss'])
    assert not torch.equal(before,model.net.input_projection.weight)
    assert torch.equal(fixed,model.fixed_embeddings)


@pytest.mark.parametrize('sequences, expected_pairs, expected_fraction', [
    ([[0,0,0]], 2, 0.),              # Staying on raw road zero needs no self-loop.
    ([[0,4,9]], 2, 0.),              # Valid directed changes.
    ([[0,9,4]], 2, 1.),              # Disconnected directed changes.
    ([[0,0,4,4,9,0,9]], 6, 1/6),    # Repeated roads remain in the denominator.
    ([[0,0],[4,4]], 2, 0.),          # Never form a pair across trajectories.
    ([[0]], 0, None),
])
def test_connectivity_observation_pair_definition(setup,sequences,expected_pairs,expected_fraction):
    from urban_syn.diffusion.conditional.standalone.runner import distribution_metrics
    _,config,_=setup
    ctx=Context(config)
    frame=pd.DataFrame({'edge_ids':sequences})
    metrics=distribution_metrics(ctx,frame,frame)
    assert metrics['transition_count']==expected_pairs
    if expected_fraction is None:
        assert metrics['invalid_transition_fraction'] is None
    else:
        assert metrics['invalid_transition_fraction']==pytest.approx(expected_fraction)
