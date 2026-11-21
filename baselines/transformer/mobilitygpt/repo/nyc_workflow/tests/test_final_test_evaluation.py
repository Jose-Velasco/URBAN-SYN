"""CPU fixtures for independent test loading and locked paired evaluation."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from nyc_workflow.paired import SEEDS, stable_seed, verify_pairing
from datasets.char_dataset import CharDataset
from nyc_workflow.data import GravityTrajectoryDataset, DeterministicTrajectoryWindows
from mobilitygpt.config_utils import CfgNode
from nyc_workflow.evaluation import (LABELS, coverage_check, eligibility, load_sections,
                                     paired_metrics, sequence_overlap, recorded_content_matches)
from nyc_workflow.paired import fingerprint


class FinalTestEvaluationTests(unittest.TestCase):
    def test_relocated_recorded_content_checks_hash_and_size_without_changing_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'artifact'
            path.write_bytes(b'preserved checkpoint identity')
            recorded = fingerprint(path)
            recorded['path'] = '/original/machine/artifact'
            self.assertTrue(recorded_content_matches(path, recorded))
            self.assertEqual(recorded['path'], '/original/machine/artifact')
            path.write_bytes(b'changed checkpoint identity')
            self.assertFalse(recorded_content_matches(path, recorded))

    def fixture(self, directory):
        nodes = [{'road_id':10,'observation_range':[0,2],'timestamp':'t0'},
                 {'road_id':20,'observation_range':None,'timestamp':''},
                 {'road_id':10,'observation_range':[2,3],'timestamp':'t2'}]
        row = {'traj_id':'test-parent:0','parent_traj_id':'test-parent','section_index':'0',
               'rid_list':'0,1,0','original_rid_list':'10,20,10','time_list':'t0,,t2',
               'observation_ranges':json.dumps([n['observation_range'] for n in nodes])}
        with (directory/'rows.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
        provenance = {'split':'test','parent_traj_id':'test-parent','nodes':nodes,
                      'sections':[{'section_index':0,'node_range':[0,3],'status':'retained',
                                   'export_traj_id':'test-parent:0'}]}
        (directory/'provenance.jsonl').write_text(json.dumps(provenance)+'\n')
        return row,provenance

    def test_test_csv_and_provenance_alignment_with_connector_and_revisit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            row,_ = self.fixture(directory)
            rows,parents,status = load_sections(directory/'rows.csv',directory/'provenance.jsonl',
                                                {10:0,20:1},{(0,1),(1,0)},'test')
            self.assertEqual(rows,[row]); self.assertEqual(parents,{'test-parent'})
            self.assertEqual(status,{'retained':1})

    def test_provenance_parent_misalignment_and_wrong_split_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            _,provenance = self.fixture(directory)
            provenance['parent_traj_id'] = 'wrong'
            (directory/'provenance.jsonl').write_text(json.dumps(provenance)+'\n')
            with self.assertRaisesRegex(ValueError,'parent or section mismatch'):
                load_sections(directory/'rows.csv',directory/'provenance.jsonl',
                              {10:0,20:1},{(0,1),(1,0)},'test')
            provenance['split'] = 'train'
            (directory/'provenance.jsonl').write_text(json.dumps(provenance)+'\n')
            with self.assertRaisesRegex(ValueError,'Wrong split'):
                load_sections(directory/'rows.csv',directory/'provenance.jsonl',
                              {10:0,20:1},{(0,1),(1,0)},'test')

    def test_connectivity_and_unknown_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp); self.fixture(directory)
            with self.assertRaisesRegex(ValueError,'Disconnected reference'):
                load_sections(directory/'rows.csv',directory/'provenance.jsonl',
                              {10:0,20:1},{(0,1)},'test')

    def test_eligibility_keeps_all_rows_and_explicit_exclusions(self):
        rows = [{'traj_id':str(i),'parent_traj_id':'same-parent','rid_list':r}
                for i,r in enumerate(('0,1','0,1,0','0,1,0,1'))]
        manifest = eligibility(rows)
        self.assertEqual(len(manifest),3)
        self.assertEqual([m['eligible_prompt_lengths'] for m in manifest],[[1],[1],[1,3]])
        self.assertEqual(manifest[0]['exclusions'],{'3':'no_recorded_continuation'})

    def test_exact_route_duplicates_do_not_mean_parent_leakage_or_removal(self):
        train = [{'traj_id':'a:0','parent_traj_id':'a','rid_list':'0,1,0'}]
        test = [{'traj_id':'b:0','parent_traj_id':'b','rid_list':'0,1,0'},
                {'traj_id':'b:1','parent_traj_id':'b','rid_list':'0,1'}]
        result = sequence_overlap(train,test)
        self.assertEqual(result['test_sections_matching_training_sequences'],1)
        self.assertEqual(result['duplicates_removed'],0)
        self.assertEqual(len(test),2)

    def test_target_coverage_includes_long_tail_first_road_eos_once(self):
        base = CharDataset(CfgNode(block_size=3),'0,1\n\n'+','.join(['0','1']*5),['0','1'])
        adapter = GravityTrajectoryDataset(base)
        windows = DeterministicTrajectoryWindows(adapter,range(len(adapter)))
        result = coverage_check(adapter,windows)
        self.assertEqual(result['targets'],14)  # 2+1 and 10+1, including EOS targets.
        self.assertEqual(result['windows'],5)
        windows.windows.append(windows.windows[-1])
        with self.assertRaisesRegex(ValueError,'Targets not covered once'):
            coverage_check(adapter,windows)

    def test_pairing_rejects_lost_case_and_paired_deltas_use_same_trajectory(self):
        selected = eligibility([{'traj_id':'t','parent_traj_id':'p','rid_list':'0,1,0,1'}])
        records = []
        for label in LABELS:
            for p in (1,3):
                for seed in SEEDS:
                    roads = selected[0]['reference_roads'][:p]
                    records.append({'checkpoint':label,'trajectory_id':'t','parent_id':'p',
                        'prompt_length':p,'prompt_roads':roads,'roads':roads,'road_count':p,
                        'seed':seed,'derived_seed':stable_seed(seed,'t',p),'immediate_stop':True,
                        'termination':'eos','distance_m':p*10,'unique_road_fraction':len(set(roads))/p,
                        'has_repeated_road':p==3})
        self.assertEqual(verify_pairing(records,selected,checkpoints=LABELS)['actual_samples'],12)
        self.assertEqual(paired_metrics(records)['1']['pooled']['road_count_delta']['mean'],0.)
        with self.assertRaisesRegex(ValueError,'Missing'):
            verify_pairing(records[:-1],selected,checkpoints=LABELS)
        bad = copy.deepcopy(records); bad[-1]['derived_seed'] += 1
        with self.assertRaisesRegex(ValueError,'Seed mismatch'):
            verify_pairing(bad,selected,checkpoints=LABELS)


if __name__ == '__main__':
    unittest.main()
