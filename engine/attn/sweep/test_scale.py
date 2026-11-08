"""Deterministic driver gates for adaptive stopping and distributed case sweeps."""
import contextlib
import importlib.util
import io
import json
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

if __name__=='__main__':unittest.main()
