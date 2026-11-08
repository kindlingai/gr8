"""Deterministic driver gates for adaptive stopping and distributed case sweeps."""
import contextlib
import importlib.util
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scale_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def fake_bundle(root, variants=('off','stable'), cases=None):
    cases=cases or [dict(id='a',args=['1'],weights={'toy':2}),dict(id='b',args=['2'],weights={'toy':3})]
    (root/'adapter.json').write_text(json.dumps(dict(name='toy',header='fixture.h',variants_dir='variants')))
    (root/'cases.json').write_text(json.dumps(cases));(root/'fixture.h').write_text('// our deterministic test adapter')
    for v in variants:
        d=root/'variants'/v;d.mkdir(parents=True);(d/'variant.cuh').write_text('// fixture')
    original=runner.command
    def command(cmd,timeout):
        if cmd[0]!='fake-nvcc':return original(cmd,timeout)
        if '--version' in cmd:return 0,'fake compiler',''
        out=Path(cmd[cmd.index('-o')+1])
        out.write_text('''#!/usr/bin/env python3
import json,pathlib,sys
p=pathlib.Path(sys.argv[0]); state=p.with_suffix('.count')
r=int(state.read_text())+1 if state.exists() else 1;state.write_text(str(r))
delta=0 if p.name=='off' else (-10 if p.name!='noisy' else (-20 if r%2 else 20))
base=100*float(sys.argv[-1])
print(json.dumps(dict(status='EXACT',mean_us=base*(1+delta/100),sd_us=1,
 reference_mean_us=base,paired_delta_pct_mean=delta,null_delta_pct_mean=10 if p.name=='badnull' else 0,
 null_band_pct=3,regime='deterministic',integrity_errors=0,launch_valid=True)))
''');out.chmod(0o755)
        return 0,'',''
    args=['--kernel',str(root/'adapter.json'),'--cases',str(root/'cases.json'),'--variants',*variants,'--nvcc','fake-nvcc','--arch','120a','--out',str(root/'bundle'),'--build-only']
    with patch.object(runner,'command',side_effect=command),contextlib.redirect_stdout(io.StringIO()):
        assert runner.main(args)==0
    return root/'bundle'

def execute(bundle, *extra):
    with patch.object(runner,'telemetry_snapshot',return_value={'status':'UNAVAILABLE','test':True}),contextlib.redirect_stdout(io.StringIO()):
        return runner.main(['--arch','120a','--out',str(bundle),'--run-only',*extra])

class AdaptiveTests(unittest.TestCase):
    def test_driver_stable_early_null_outside_and_noisy_hit_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=fake_bundle(root,('off','stable','noisy','badnull'),[dict(id='a',args=['1'],weights={})])
            self.assertEqual(execute(bundle,'--adaptive-ci','1'),0)
            decisions=[json.loads(x) for x in (bundle/'adaptive-decisions.jsonl').read_text().splitlines()]
            final={d['variant']:d for d in decisions}
            self.assertEqual(final['stable']['rounds_used'],3)
            self.assertEqual(final['stable']['adaptive_stop_reason'],'CI_REACHED')
            self.assertEqual(final['stable']['final_ci95_half_width_pct'],0)
            self.assertEqual(final['noisy']['rounds_used'],6)
            self.assertEqual(final['noisy']['adaptive_stop_reason'],'MAX_ROUNDS_CI_WIDE')
            self.assertEqual(final['badnull']['rounds_used'],6)
            self.assertEqual(final['badnull']['adaptive_stop_reason'],'MAX_ROUNDS_NULL_OUTSIDE')
            self.assertIn('final_ci95_half_width_pct',(bundle/'summary.csv').read_text())

    def test_null_gate_sticky_missing_and_invalid(self):
        raw=[];pairs=[]
        for r in range(1,7):
            rid=f'a.r{r}'
            raw.append(dict(run_id=rid,final_attempt=True,null_delta_pct_mean=4 if r==1 else 0,null_band_pct=3))
            pairs.append(dict(case='a',variant='v',round_id=r,status='EXACT',delta_pct=-10,estimator='paired',regime='x',run_ids=[rid]))
            d=runner.adaptive_decision(raw,pairs,'a','v',1)
            self.assertEqual(d['stop'],r==6)
        self.assertEqual(d['adaptive_stop_reason'],'MAX_ROUNDS_NULL_OUTSIDE')
        raw[0].pop('null_delta_pct_mean');d=runner.adaptive_decision(raw,pairs,'a','v',1)
        self.assertEqual(d['adaptive_stop_reason'],'MAX_ROUNDS_NULL_UNAVAILABLE')
        raw[0]['null_delta_pct_mean']=0;pairs[0]['status']='INVALID_PAIR'
        self.assertEqual(runner.adaptive_decision(raw,pairs,'a','v',1)['adaptive_stop_reason'],'MAX_ROUNDS_INVALID')

    def test_default_fixed_rounding_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle=fake_bundle(Path(tmp),('off','stable','noisy'),[dict(id='a',args=['1'],weights={})])
            self.assertEqual(execute(bundle),0)
            pairs=[json.loads(x) for x in (bundle/'pairs.jsonl').read_text().splitlines()]
            self.assertEqual(len(pairs),8) # two candidates, complete balanced four rounds
            self.assertFalse((bundle/'adaptive-decisions.jsonl').exists())

class ReducedTests(unittest.TestCase):
    def test_union17_and_coverage_is_archived_time_not_call_count(self):
        cases=runner.validate_cases(json.loads((HERE/'cases_reduced_r2.json').read_text()))
        self.assertEqual(len(cases),17)
        self.assertEqual(sum(bool(c['weights']['32K']) for c in cases),4)
        self.assertEqual(sum(bool(c['weights']['128K']) for c in cases),16)
        self.assertTrue(all(c['args'][2]==2 and c['args'][0] in (8191,8192) for c in cases))
        self.assertTrue(any(c['args'][:3]==[8191,24576,2] for c in cases))
        coverage=runner.reduced_coverage(cases)
        self.assertAlmostEqual(coverage['32K'],90.26980267364321)
        self.assertAlmostEqual(coverage['128K'],90.47806185875925)
        rows=[dict(kernel='attention_cq',variant='off',case=c['id'],status='EXACT',mean_us=c['baseline_off_mean_us'],sd_us=0) for c in cases]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);tables=runner.write_tables(out,rows,cases,['off'],'off','attention_cq')
            totals=[r for r in tables if r['case'].startswith('TOTAL')]
            self.assertTrue(all(r['scope']=='reduced_set_kernel_only' for r in totals))
            self.assertIn('reduced set: 90.2698 % of 32K / 90.4781 % of 128K off weight',(out/'summary.md').read_text())
            self.assertIn('NOT full prompt/TTFT',(out/'summary.md').read_text())
        bad=[dict(cases[0]),dict(cases[1])];bad[1].pop('reduced_set')
        with self.assertRaises(ValueError):runner.reduced_coverage(bad)

class ShardTests(unittest.TestCase):
    def test_real_driver_mock_shards_merge_equal_unsharded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=fake_bundle(root)
            copies=[root/'unsharded',root/'s0',root/'s1']
            for d in copies:shutil.copytree(original,d)
            self.assertEqual(execute(copies[0],'--adaptive-ci','1'),0)
            for i in range(2):self.assertEqual(execute(copies[i+1],'--adaptive-ci','1','--shard',f'{i}/2'),0)
            self.assertEqual(runner.merge_runs(copies[1:],root/'merged'),0)
            self.assertEqual((root/'merged/summary.csv').read_bytes(),(copies[0]/'summary.csv').read_bytes())
            self.assertEqual((root/'merged/summary.md').read_bytes(),(copies[0]/'summary.md').read_bytes())
            self.assertTrue((root/'merged/shards/0/results.jsonl').exists())
            self.assertTrue((original/'merge.py').exists())
            with self.assertRaises(ValueError):runner.merge_runs([copies[1]],root/'partial')
            with self.assertRaises(ValueError):runner.merge_runs([copies[1],copies[1]],root/'duplicate')
            m=json.loads((copies[2]/'run-manifest.json').read_text());m['seed']+=1
            (copies[2]/'run-manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):runner.merge_runs(copies[1:],root/'wrong-settings')
            m['seed']-=1;m.pop('finished_unix');(copies[2]/'run-manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):runner.merge_runs(copies[1:],root/'unfinished')

    def test_shard_partition_validation(self):
        cases=[dict(id=str(i),args=[]) for i in range(17)]
        shards=[runner.shard_cases(cases,f'{i}/4')[0] for i in range(4)]
        self.assertEqual([len(x) for x in shards],[5,4,4,4])
        self.assertEqual({c['id'] for s in shards for c in s},{c['id'] for c in cases})
        for bad in ('1/0','4/4','-1/4','0/18','0','x/4'):
            with self.assertRaises(ValueError):runner.shard_cases(cases,bad)

if __name__=='__main__':unittest.main()
