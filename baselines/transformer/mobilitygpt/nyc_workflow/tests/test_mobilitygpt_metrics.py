"""Deterministic parity against frozen pure notebook definitions.

Neither the notebook nor its data-loading/plotting/evaluation cells are required.
Run: .venv/bin/python -m unittest discover -s tests -p test_mobilitygpt_metrics.py -v
"""
import ast
from collections import Counter, defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import entropy
from pyproj import Transformer
from shapely import STRtree, points as shapely_points

from nyc_workflow import metrics as m

ROOT = Path(__file__).resolve().parents[1]


def notebook_functions():
    """Independent frozen definitions; complete notebook is archived."""
    from nyc_workflow.tests import notebook_metrics_reference
    return vars(notebook_metrics_reference)


def write_csv(path, rows, columns=None):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_fixture(root):
    """Complete tiny artifact layout, including one-to-many explicit provenance."""
    run, raw = root/'run', root/'raw'
    prepared = run/'prepared/NYCAll-Taxi'
    prepared.mkdir(parents=True)
    raw.mkdir()
    (run/'evaluation').mkdir()
    geo_rows = [dict(geo_id=i, coordinates=json.dumps([[-74+i*.001, 40.7], [-74+i*.001, 40.702]])) for i in (10, 20, 30)]
    write_csv(raw/'nyc.geo', geo_rows)
    write_csv(prepared/'roadmap.geo', [{**r, 'geo_id': i} for i,r in enumerate(geo_rows)])
    write_csv(raw/'nyc.rel', [dict(rel_id=0, origin_id=10, destination_id=20),
                              dict(rel_id=1, origin_id=20, destination_id=30)])
    write_csv(raw/'nyc_mm_train.csv', [dict(traj_id='member', rid_list='10,10,20,30')])
    write_csv(raw/'nyc_mm_test.csv', [dict(traj_id='parent:unusual', rid_list='10,10,20,30'),
                                     dict(traj_id='other', rid_list='30,30')])
    write_csv(prepared/'id_mapping.csv', [dict(original_id=i, new_id=j) for j,i in enumerate((10,20,30))])
    rows, sections = [], []
    nodes = [dict(road_id=10, observation_range=[0,2], timestamp='a'),
             dict(road_id=20, observation_range=[2,3], timestamp='b'),
             dict(road_id=30, observation_range=[3,4], timestamp='c')]
    for sid, a, b in [('arbitrary-section-A',0,2), ('unrelated-section-name',2,3)]:
        selected = nodes[a:b]
        index = len(rows)
        rows.append(dict(traj_id=sid, parent_traj_id='parent:unusual', section_index=index,
                         rid_list=','.join(str((n['road_id']//10)-1) for n in selected),
                         original_rid_list=','.join(str(n['road_id']) for n in selected),
                         time_list=','.join(n['timestamp'] for n in selected),
                         observation_ranges=json.dumps([n['observation_range'] for n in selected])))
        sections.append(dict(section_index=index, node_range=[a,b], status='retained', export_traj_id=sid))
    # For 2 distinct prompt lengths with eligible data, make one whole 3-road
    # section and one singleton belonging to the same parent. Overlapping ranges
    # would be invalid: extend the raw parent instead.
    write_csv(raw/'nyc_mm_test.csv', [dict(traj_id='parent:unusual', rid_list='10,10,20,30,10'),
                                     dict(traj_id='other', rid_list='30,30')])
    nodes.append(dict(road_id=10, observation_range=[4,5], timestamp='d'))
    rows[0].update(rid_list='0,1,2', original_rid_list='10,20,30', time_list='a,b,c',
                   observation_ranges='[[0,2],[2,3],[3,4]]')
    rows[1].update(rid_list='0', original_rid_list='10', time_list='d', observation_ranges='[[4,5]]')
    sections[0]['node_range'], sections[1]['node_range'] = [0,3], [3,4]
    write_csv(prepared/'NYCAll_Taxi_trajectory_test.csv', rows)
    trip = dict(split='test', parent_traj_id='parent:unusual', original_observations=5, nodes=nodes, sections=sections)
    (prepared/'test_provenance.jsonl').write_text(json.dumps(trip)+'\n')
    config = dict(dpo=dict(selected_update=100), evaluation=dict(prompt_roads=[1,2], seeds=[42,43,44]))
    (run/'pipeline_config.json').write_text(json.dumps(config))
    sources = {name: m.fingerprint(raw/file) for name,file in
               [('geography','nyc.geo'), ('relations','nyc.rel'), ('train','nyc_mm_train.csv'), ('test','nyc_mm_test.csv')]}
    (prepared/'processing_metadata.json').write_text(json.dumps(dict(source_files=sources)))
    records = []
    for cp in ['pretrained','selected_dpo_100']:
        for prompt in [1,2]:
            for seed in [42,43,44]:
                records.append(dict(checkpoint=cp, prompt_length=prompt, seed=seed,
                    trajectory_id=rows[0]['traj_id'], parent_id='parent:unusual', trajectory_index=0,
                    roads=[0,1,1] if prompt == 2 else [0], prompt_roads=list(range(prompt)),
                    termination='cap' if prompt == 2 else 'eos', immediate_stop=prompt == 1,
                    road_count=3 if prompt == 2 else 1))
    (run/'evaluation/generated_trajectories.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return run, raw, prepared


class NotebookParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = notebook_functions()
        cls.a = np.array([[40.7,-74], [40.7,-74], [40.704,-73.99], [40.71,-73.98]])
        cls.b = np.array([[40.7,-74], [40.706,-73.99], [40.709,-73.98]])

    def test_constants_and_haversine(self):
        for name in ['EARTH_RADIUS_KM','GRID_LON_MIN','GRID_LAT_MIN','GRID_WIDTH','GRID_HEIGHT',
                     'DISTANCE_MAX_KM','RADIUS_MAX_KM','EDR_EPS_M','FASTDTW_RADIUS','PRIVACY_POINTS']:
            self.assertEqual(getattr(m,name), self.nb[name])
        np.testing.assert_array_equal(m.pairwise_haversine_matrix_km(self.a,self.b),
                                     self.nb['pairwise_haversine_matrix_km'](self.a,self.b))

    def test_all_distributions(self):
        real, gen = {'a':self.a, 'b':self.b}, {'c':self.b, 'd':self.b[:1], 'e':self.b}
        expected, _ = self.nb['compute_distribution_metrics'](real,gen)
        actual = m.distribution_metrics(m.distribution_summary(real)[0], m.distribution_summary(gen)[0])
        self.assertEqual(actual, expected)
        for name in ('trip_distance_km', 'ts_style_radius_km'):
            for t in (self.a,self.b,self.a[:1],self.a[:0]):
                self.assertEqual(getattr(m,name)(t), self.nb[name](t))

    def test_four_pairwise_functions(self):
        for a,b in [(self.a,self.b),(self.a[:1],self.b),(self.a,self.b[:1])]:
            np.testing.assert_array_equal(m.compute_single_pair_metrics(a,b),
                                         self.nb['compute_single_pair_metrics'](a,b))
        self.assertIsNone(m.compute_single_pair_metrics(self.a[:0], self.b))
        self.assertTrue(math.isnan(m.edr_distance(self.a[:0], self.b)))

    def test_privacy_resampling_float32_and_auc(self):
        for t in (self.a,self.b,self.a[:1],self.a[:0],np.repeat(self.a[:1],5,axis=0)):
            np.testing.assert_array_equal(m.resample_trajectory(t), self.nb['resample_trajectory'](t,32))
        arrays = np.stack([m.resample_trajectory(self.a), m.resample_trajectory(self.b)])
        distances = m.matched_point_trajectory_distance_matrix(arrays, arrays, 1)
        self.assertEqual(distances.dtype, np.float32)
        np.testing.assert_array_equal(distances, self.nb['matched_point_trajectory_distance_matrix'](arrays,arrays,1))
        many=arrays[np.arange(35)%2]
        np.testing.assert_array_equal(m.matched_point_trajectory_distance_matrix(many,arrays),
                                     self.nb['matched_point_trajectory_distance_matrix'](many,arrays))
        for pos, neg in [([0,0],[0,0]), ([1,2],[0,0]), ([0,1],[1,2])]:
            self.assertEqual(m.rank_auc(pos,neg),self.nb['rank_auc'](pos,neg))
        train, test, gen = {'member':self.a}, {'nonmember':self.b}, {'generated':self.a}
        expected, _ = self.nb['compute_privacy_metrics'](train,test,gen)
        actual, _ = m.privacy_metrics(train,test,gen,{'generated':'different-parent'})
        self.assertEqual(actual,expected)

    def test_sampling(self):
        lookup = {str(i):self.a for i in range(2010)}
        for seed in [43,44,45]:
            self.assertEqual(list(m.sample_lookup(lookup,2000,seed)), list(self.nb['sample_lookup'](lookup,2000,seed)))
        pairs = [(str(i),str(i)) for i in range(5010)]
        self.assertEqual(m.choose_pairs(pairs),self.nb['choose_pairs'](pairs,5000,42))

    def test_rsc_notebook_parity(self):
        routes={'a':[1,2,2,3], 'b':[3,2], 'c':[1]}
        self.assertEqual(m.compute_rsc_from_geo_rel(routes,{(1,2),(2,3)}),
                         self.nb['compute_rsc_from_geo_rel'](routes,{(1,2),(2,3)}))


class Boundaries(unittest.TestCase):
    def test_jsd_and_overflow(self):
        self.assertEqual(m.js_divergence([1,0],[1,0]),0)
        self.assertAlmostEqual(m.js_divergence([1,0],[0,1]),math.log(2))
        self.assertTrue(math.isnan(m.js_divergence([0,0],[1,0])))
        # Raw counts mixed BEFORE entropy normalization, a material notebook detail.
        p,q = np.array([2.,0]),np.array([0.,1])
        self.assertEqual(m.js_divergence(p,q),.5*entropy(p,(p+q)/2)+.5*entropy(q,(p+q)/2))
        for edges,bound,count in [(m.DISTANCE_EDGES,40.71,1000),(m.RADIUS_EDGES,10.6,100)]:
            self.assertEqual(len(edges), count+2)
            hist = np.histogram([0,bound-1e-8,bound,bound+10],bins=edges)[0]
            self.assertEqual(hist[-1],2)
            self.assertEqual(hist[-2],1)

    def test_grid_bounds_and_plotting_distinction(self):
        lat,lon = m.GRID_LAT_MIN,m.GRID_LON_MIN
        self.assertEqual(m.point_to_grid(lat,lon),0)
        self.assertIsNone(m.point_to_grid(lat-1e-10,lon))
        self.assertIsNone(m.point_to_grid(lat,lon-1e-10))
        self.assertEqual(m.point_to_grid(lat+.449999,lon+.434999),7829)
        self.assertIsNone(m.point_to_grid(lat+.450001,lon))
        self.assertIsNone(m.point_to_grid(lat,lon+.435001))
        self.assertIsNotNone(m.point_to_grid(m.GRID_LAT_MAX+.001,m.GRID_LON_MAX+.001))
        t = np.array([[lat,lon],[lat,lon],[lat-1,lon]])
        counts,meta=m.grid_distribution({'t':t})
        self.assertEqual(counts[0],2)
        self.assertEqual(meta['oob_points'],1)
        self.assertEqual(m.od_distribution({'t':t})[1]['oob_od_trajectories'],1)
        self.assertTrue(math.isnan(m.grid_jaccard(t[-1:],t[-1:])))

    def test_strict_edr_boundary_and_dtw_normalization(self):
        a,b=np.array([[0.,0.]]),np.array([[0.,.001]])
        exact=m.haversine_km_pair(a[0],b[0])*1000
        self.assertEqual(m.edr_distance(a,b,exact),1)
        self.assertEqual(m.edr_distance(a,b,np.nextafter(exact,np.inf)),0)
        self.assertEqual(m.edr_distance(a,b,np.nextafter(exact,-np.inf)),1)
        with patch('fastdtw.fastdtw',return_value=(12.,[(0,0),(1,0),(2,0)])) as call:
            self.assertEqual(m.normalized_dtw_km(a,b),4)
            self.assertEqual(call.call_args.kwargs['radius'],1)
        with patch('fastdtw.fastdtw',return_value=(0.,[])):
            self.assertTrue(math.isnan(m.normalized_dtw_km(a,b)))

    def test_pairwise_nanmean_and_empty_pair_counts(self):
        outside=np.array([[0.,0.]])
        inside=np.array([[40.7,-74.]])
        real={'a':outside,'b':inside,'empty':np.empty((0,2))}
        gen={'x':outside,'y':inside,'z':inside}
        metrics,diag=m.pairwise_metrics(real,gen,{'x':'a','y':'b','z':'empty'},1)
        self.assertEqual(metrics['jaccard'],1)
        self.assertEqual(diag['finite_counts']['jaccard'],1)
        self.assertEqual(diag['pair_count'],2)
        self.assertEqual(diag['skipped_empty'],1)

    def test_privacy_parent_exclusions_and_identical_other_route(self):
        t=np.array([[40.7,-74.]])
        members,nonmembers={'member':t},{'parent':t}
        generated={'section:a':t,'section:b':t,'other':t}
        values,diag=m.privacy_metrics(members,nonmembers,generated,
                                     {'section:a':'parent','section:b':'parent','other':'other-parent'})
        self.assertEqual(values['mia_auc'],.5)
        self.assertEqual(diag['excluded_comparisons'],2)
        self.assertEqual(values['uniqueness'],0)
        values,diag=m.privacy_metrics(members,nonmembers,{'section:a':t,'section:b':t},
                                     {'section:a':'parent','section:b':'parent'})
        self.assertTrue(math.isnan(values['mia_auc']))
        self.assertEqual(diag['unavailable_test_score_ids'],['parent'])
        self.assertEqual(diag['nonmember_scores'],[None])
        self.assertEqual(m.rank_auc([1,1],[1,1]),.5)

    def test_directed_reversed_self_singleton_rsc(self):
        rate,diag=m.compute_rsc_from_geo_rel({'forward':[10,20], 'reverse':[20,10],
                                            'self':[30,30], 'singleton':[10]}, {(10,20)})
        self.assertEqual(rate,2/3)
        self.assertEqual(diag,dict(valid_transitions=2,total_transitions=3))
        self.assertTrue(math.isnan(m.compute_rsc_from_geo_rel({'x':[10]},set())[0]))

    def test_privacy_uniqueness_strict_threshold_and_always_resample(self):
        t=np.c_[np.full(32,40.7),-74+np.linspace(0,.01,32)**2]
        distance=np.array([[.05,np.nextafter(np.float32(.05),np.float32(np.inf))]],dtype=np.float32)
        with patch.object(m,'matched_point_trajectory_distance_matrix',return_value=distance.copy()), \
             patch.object(m,'resample_trajectory',wraps=m.resample_trajectory) as resample:
            values,_=m.privacy_metrics({'member':t},{'test':t},{'g1':t,'g2':t}, {'g1':'other','g2':'other'})
            self.assertEqual(values['uniqueness'],.5)
            self.assertEqual(resample.call_count,4)
        with self.assertRaisesRegex(ValueError,'empty trajectories'):
            m.privacy_metrics({'member':t[:0]},{'test':t},{'g':t},{'g':'other'})

    def test_off_network_strict_threshold(self):
        class FakeTree:
            def query_nearest(self, *args, **kwargs):
                return np.array([[0,1,2],[0,0,0]]),np.array([30.,np.nextafter(30.,np.inf),29.])
        class FakeTransformer:
            def transform(self,x,y):
                return x,y
        rate,diag=m.compute_off_network_rate({'t':np.array([[40.,-74.]]*3)},FakeTree(),FakeTransformer())
        self.assertEqual(rate,1/3)
        self.assertEqual(diag['off_network_point_count'],1)

    def test_sequential_parallel_agreement(self):
        t=np.array([[40.7,-74.],[40.7,-74.],[40.71,-74.]])
        real={'parent':t}
        gen={'a':t[:1],'b':t,'c':t[:2]}
        parents={k:'parent' for k in gen}
        environment=dict(os.environ)
        seq,sd=m.pairwise_metrics(real,gen,parents,1)
        parallel,pd=m.pairwise_metrics(real,gen,parents,2)
        self.assertEqual(seq,parallel)
        self.assertEqual(sd['selected_pairs'],pd['selected_pairs'])
        self.assertEqual(sd['pair_count'],3)
        self.assertEqual(environment,dict(os.environ))


class ArtifactAndNetworkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.run,self.raw,self.prepared=build_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def args(self):
        from argparse import Namespace
        return Namespace(run_dir=self.run,raw_data_dir=self.raw,notebook=None,canonical_network=None)

    def test_load_preserves_ids_latlon_repeats_singletons_and_hashes(self):
        before=m.hashes([p for p in self.root.rglob('*') if p.is_file()])
        train,test,groups,adj,diag,hashes,config=m.load_inputs(self.args())
        self.assertEqual(len(groups),12)
        self.assertEqual(test['parent:unusual'].shape,(5,2))
        np.testing.assert_array_equal(test['parent:unusual'][0],test['parent:unusual'][1])
        np.testing.assert_array_equal(test['parent:unusual'][0],[40.701,-73.99])
        self.assertEqual(len(groups['pretrained',1,42]['gps']['arbitrary-section-A']),1)
        self.assertEqual(groups['pretrained',2,42]['roads']['arbitrary-section-A'],[10,20,20])
        for p in [1,2]:
            g1,g2=groups['pretrained',p,42],groups['selected_dpo_100',p,42]
            self.assertEqual(list(m.sample_lookup(g1['gps'],1,45)),list(m.sample_lookup(g2['gps'],1,45)))
        self.assertEqual(before,m.hashes([Path(p) for p in before]))
        sections=m.load_sections(self.prepared/'NYCAll_Taxi_trajectory_test.csv',self.prepared/'test_provenance.jsonl',
                                 {0:10,1:20,2:30},m.load_road_lookup(self.raw/'nyc_mm_test.csv'))
        self.assertEqual([s['parent'] for s in sections.values()],['parent:unusual']*2)

    def test_missing_parent_no_heuristic_fallback(self):
        test=m.load_road_lookup(self.raw/'nyc_mm_test.csv')
        del test['parent:unusual']
        with self.assertRaisesRegex(ValueError,'Missing raw parent'):
            m.load_sections(self.prepared/'NYCAll_Taxi_trajectory_test.csv',self.prepared/'test_provenance.jsonl',{0:10,1:20,2:30},test)

    def test_duplicate_keys_and_inconsistent_mappings(self):
        path=self.run/'evaluation/generated_trajectories.jsonl'
        original=path.read_text()
        path.write_text(original+original.splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'Duplicate generated record'):
            m.load_inputs(self.args())
        path.write_text(original)
        mapping=self.prepared/'id_mapping.csv'
        mapping.write_text(mapping.read_text()+'10,0\n')
        with self.assertRaisesRegex(ValueError,'inconsistent road mapping'):
            m.load_inputs(self.args())

    def test_unknown_ids_and_nonfinite_geometry(self):
        with self.assertRaisesRegex(ValueError,'Unknown original road'):
            m.road_lookup_to_gps({'t':[99]}, {10:(40.,-74.)})
        path=self.raw/'bad.geo'
        write_csv(path,[dict(geo_id=0,coordinates='[[1e309,40],[-74,40]]')])
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            m.load_geo_id_to_latlon(path)

    def test_unknown_generated_id_and_changed_raw_hash(self):
        path=self.run/'evaluation/generated_trajectories.jsonl'
        records=list(m.jsonl(path))
        records[0]['roads']=[0,999]
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
        with self.assertRaisesRegex(ValueError,'Unknown generated road'):
            m.load_inputs(self.args())
        raw=self.raw/'nyc_mm_test.csv'
        raw.write_text(raw.read_text()+'other,30\n')
        with self.assertRaisesRegex(ValueError,'provenance hash'):
            m.load_inputs(self.args())

    def test_network_crs_and_projected_nearest(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        missing=gpd.GeoDataFrame(geometry=[LineString([(-74,40.7),(-74,40.71)])])
        with patch('geopandas.read_file',return_value=missing):
            with self.assertRaisesRegex(ValueError,'no CRS'):
                m.load_road_spatial_index('unused')
        roads=missing.set_crs(4326)
        roads['drive']=['yes']
        with patch('geopandas.read_file',return_value=roads):
            tree,transform,diag=m.load_road_spatial_index('unused',mode_column='drive')
            rate,coverage=m.compute_off_network_rate({'t':np.array([[40.705,-74],[40.705,-73.99]])},tree,transform)
            self.assertEqual(rate,.5)
            self.assertEqual(coverage['generated_point_count'],2)
            self.assertIn('32618',diag['metric_crs'])
            with self.assertRaisesRegex(ValueError,'Missing network mode'):
                m.load_road_spatial_index('unused',mode_column='walk')
            notebook=notebook_functions()
            expected,_=notebook['compute_off_network_rate']({'t':np.array([[40.705,-74],[40.705,-73.99]])},tree,transform)
            self.assertEqual(rate,expected)

    def test_cli_validate_refusal_strict_json_and_full_fixture(self):
        output=self.root/'validation'
        argv=['--run-dir',str(self.run),'--raw-data-dir',str(self.raw),'--allow-missing-network',
              '--output-dir',str(output),'--jobs','1']
        with patch('sys.stdout',new=io.StringIO()):
            m.main(argv+['--validate-only'])
        manifest=json.loads((output/'manifest.json').read_text())
        self.assertTrue(manifest['input_hashes_preserved'])
        self.assertFalse((output/'metrics.csv').exists())
        with self.assertRaisesRegex(ValueError,'existing output'):
            m.main(argv)
        argv[argv.index(str(output))]=str(self.root/'full')
        with patch('sys.stdout',new=io.StringIO()):
            m.main(argv)
        strict=json.loads((self.root/'full/results.json').read_text(),parse_constant=lambda x: self.fail(x))
        self.assertEqual(len(strict['results']),12)
        self.assertFalse(strict['complete'])
        self.assertEqual(strict['status'],'incomplete')
        self.assertIsNone(strict['results'][0]['off_network_rate'])
        self.assertIn('off_network_rate',strict['results'][0]['unavailable_reasons'])
        self.assertTrue(strict['input_hashes_preserved'])
        with (self.root/'full/metrics.csv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f))),12)

    def test_cli_network_required_and_benchmark_only(self):
        output=self.root/'output'
        args=['--run-dir',str(self.run),'--raw-data-dir',str(self.raw),'--output-dir',str(output)]
        with self.assertRaisesRegex(ValueError,'Canonical network required'):
            m.main(args)
        self.assertFalse(output.exists())
        with patch('sys.stdout',new=io.StringIO()):
            m.main(args+['--allow-missing-network','--benchmark-only'])
        result=json.loads((output/'manifest.json').read_text())
        self.assertEqual(result['status'],'benchmark_only')
        self.assertFalse((output/'metrics.csv').exists())
        self.assertTrue(result['input_hashes_preserved'])

    def test_run_owned_metrics_output_and_supplied_missing_network(self):
        output = self.run/'canonical_metrics'
        args = ['--run-dir', str(self.run), '--raw-data-dir', str(self.raw),
                '--output-dir', str(output), '--jobs', '1', '--allow-missing-network']
        with self.assertRaisesRegex(ValueError, 'Supplied canonical network is not a file'):
            m.main(args+['--canonical-network', str(self.root/'absent.gpkg')])
        self.assertFalse(output.exists())
        with patch('sys.stdout', new=io.StringIO()):
            m.main(args)
        result = json.loads((output/'results.json').read_text())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(len(result['results']), 12)
        self.assertTrue(all(row['off_network_rate'] is None for row in result['results']))
        with self.assertRaisesRegex(ValueError, 'separate'):
            m.main(['--run-dir', str(self.run), '--raw-data-dir', str(self.raw),
                    '--output-dir', str(self.run/'prepared/overwrite'), '--allow-missing-network'])

    def test_full_canonical_network_and_crs_failure_before_output(self):
        import geopandas as gpd
        from shapely.geometry import LineString
        network=self.root/'canonical.gpkg'
        network.write_bytes(b'fixture: read_file supplies declared canonical geometry')
        output=self.root/'canonical_output'
        args=['--run-dir',str(self.run),'--raw-data-dir',str(self.raw),'--output-dir',str(output),
              '--canonical-network',str(network),'--jobs','1']
        roads=gpd.GeoDataFrame(geometry=[LineString([(-73.99,40.69),(-73.99,40.72)])])
        with patch('geopandas.read_file',return_value=roads):
            with self.assertRaisesRegex(ValueError,'no CRS'):
                m.main(args+['--allow-missing-network'])
        self.assertFalse(output.exists())
        with patch('geopandas.read_file',return_value=roads.set_crs(4326)),patch('sys.stdout',new=io.StringIO()):
            m.main(args)
        results=json.loads((output/'results.json').read_text())
        self.assertTrue(results['complete'])
        self.assertEqual(results['status'],'complete')
        self.assertEqual(len(results['results']),12)
        self.assertTrue(all(r['off_network_rate'] is not None for r in results['results']))
        self.assertEqual(results['input_hashes_before'][str(network)],m.fingerprint(network))


if __name__=='__main__':
    unittest.main()
