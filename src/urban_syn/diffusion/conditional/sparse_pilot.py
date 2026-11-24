"""Reproducible synthetic compatibility smoke for the canonical joint GNN model.

This creates fresh artifacts and compares enabled/disabled training and known/
masked generation. It makes no claim about conditional quality on real data.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data

from .conditioning import SparseMetadataEncoder, encode_requests
from .prepared_metadata import build_prepared_metadata
from .sparse_metadata import fit_fields
from ..config import JointTrainingSettings
from ..training import build_joint_model
from ..sampling.ddpm import DDPMSampler


def smoke_inputs(output):
    """Small user-disjoint prepared fixture containing raw road zero."""
    output.mkdir(parents=True)
    train = pd.DataFrame([dict(fmm_id=i, edge_ids=[0, 4, 9], rates=[.1,.5,.9], num_points=3) for i in range(1, 5)])
    test = train.iloc[:1].assign(fmm_id=5)
    mapping = pd.DataFrame([dict(fmm_id=i, uid=str(i), tid='t', move_id='m') for i in range(1, 6)])
    observations = pd.DataFrame([dict(uid=str(i), tid='t', move_id='m', datetime=1704067200+j,
                                      label='walk' if i % 2 else 'bus') for i in range(1,6) for j in range(3)])
    weather = pd.DataFrame([dict(uid=str(i), tid='t', DATE='2024-01-01',
                                DESCRIPTION='sunny' if i % 2 else 'rain', TAVG_C=float(i*5)) for i in range(1,6)])
    for name, frame in [('train',train),('test',test),('observations', observations),('weather',weather)]:
        frame.to_parquet(output/f'{name}.parquet', index=False)
    mapping.to_csv(output/'mapping.csv', index=False)
    return {name: output/f'{name}.parquet' for name in ['train','test','observations','weather']} | {'mapping':output/'mapping.csv'}


def run_sparse_pilot(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh smoke directory: {output}')
    torch.set_num_threads(2)
    paths = smoke_inputs(output/'inputs')
    manifest = build_prepared_metadata(**paths, output=output/'metadata')
    rows = pd.read_parquet(output/'metadata/gen_train.parquet')
    # Row 4 stands in for validation; its values never fit preprocessing.
    fields = fit_fields(rows.iloc[:3], manifest['fields'])
    architecture = dict(condition_dim=8, all_dropout=0., field_dropout=0.)
    schema = dict(version=3, fields=fields, provenance=manifest, architecture=architecture,
                  train_fmm_ids=[1,2,3], validation_fmm_ids=[4])
    (output/'fitted_manifest.json').write_text(json.dumps(schema, indent=2)+'\n')
    graph = Data(x=torch.tensor([[1.,0.],[0.,1.],[1.,1.]]), edge_id=torch.tensor([0,4,9]),
                 edge_index=torch.tensor([[0,1,2,1],[1,2,0,0]]))
    settings = JointTrainingSettings.model_validate(dict(gnn=dict(hidden_dim=8, output_dim=4, num_layers=1, dropout=0.),
             diffusion=dict(hidden_dim=8, residual_layers=1, diffusion_steps=3), training=dict(epochs=1)))
    ids = torch.tensor([[1,2,3],[1,2,3],[1,2,3]])
    rates = torch.tensor([[-.8,0.,.8]]*3)
    metadata = encode_requests(rows.iloc[:3][manifest['fields']].to_dict('records'), fields)
    report = {'synthetic':True, 'runs':{}}
    for name, enabled, frozen in [('disabled',False,False),('enabled',True,False),('frozen',True,True)]:
        torch.manual_seed(42)
        encoder = SparseMetadataEncoder(fields, **architecture) if enabled else None
        model = build_joint_model(graph, settings, metadata_encoder=encoder)
        model.metadata_schema = schema if enabled else None
        model.road_encoder.requires_grad_(not frozen)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        before = {k:v.clone() for k,v in model.road_encoder.state_dict().items()}
        # Reset forward RNG so noising and timesteps match across comparisons.
        torch.manual_seed(100)
        losses = model(graph, ids, rates, metadata=metadata if enabled else None)
        loss = losses[0] + losses[2]
        assert torch.isfinite(loss)
        loss.backward()
        gnn_grad = sum(p.grad.norm().item() for p in model.road_encoder.parameters() if p.grad is not None)
        metadata_grad = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None) if enabled else 0.
        assert gnn_grad > 0 if not frozen else gnn_grad == 0
        assert not enabled or metadata_grad > 0
        optimizer.step()
        if frozen:
            assert all(torch.equal(v, model.road_encoder.state_dict()[k]) for k,v in before.items())
        model.eval()
        noisy, steps = torch.randn(3,3,5), torch.zeros(3,1,dtype=torch.long)
        null = encode_requests([{}]*3, fields)
        assert torch.equal(model.diff_model(noisy, steps), model.diff_model(noisy, steps, metadata=null if enabled else None))
        checkpoint = output/f'{name}.pt'
        torch.save(dict(config=settings.model_dump(mode='json'), model_state_dict=model.state_dict(),
                        metadata_schema=model.metadata_schema, optimizer_state_dict=optimizer.state_dict()), checkpoint)
        from ..generate import load_generation_model
        restored = load_generation_model(checkpoint, graph, settings, torch.device('cpu'))
        assert torch.equal(model.diff_model(noisy,steps), restored.diff_model(noisy,steps))
        result = dict(loss=loss.item(), gnn_gradient=gnn_grad, metadata_gradient=metadata_grad)
        if enabled:
            def sample(meta):
                torch.manual_seed(901)
                return restored.generate_data(graph,3,3,DDPMSampler(),metadata=meta)
            known, masked = sample(metadata), sample(null)
            assert (known[0]>0).all() and (masked[0]>0).all()
            # Check continuous reverse trajectories, since road argmax can coincide.
            def reverse(meta):
                torch.manual_seed(901)
                return DDPMSampler().sample(restored.diff_model, restored.diffusion_param,3,3,5,metadata=meta)
            delta = (reverse(metadata)-reverse(null)).abs().max().item()
            assert delta > 0
            result['known_masked_max_delta'] = delta
        report['runs'][name] = result
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run_sparse_pilot(parser.parse_args().output), indent=2))


if __name__ == '__main__':
    main()
