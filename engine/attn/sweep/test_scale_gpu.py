#!/usr/bin/env python3
"""Opt-in 5090 adaptive/shard integration gate; all GPU work shares the lock.
Real repeated timings need not equal. Exact merge equivalence is checked by
partitioning the SAME measured two-case observations; separate real shard runs
check dispatch/coverage/status, not invented bitwise timing reproducibility.
"""
import argparse
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
from run import cases_hash, merge_runs

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    cases=[dict(id=f'repro_r{ratio}',args=[8192,24576,ratio,0,128,32768],weights={'smoke':1}) for ratio in (2,0)]
    (root/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    bundle=root/'build'
    subprocess.run([sys.executable,str(HERE/'run.py'),'--arch','120a','--variants','off','pipe','--cases',str(root/'cases.json'),'--build-only','--out',str(bundle)],check=True)
    destinations=[root/'unsharded',root/'shard0',root/'shard1']
    for dest in destinations:shutil.copytree(bundle,dest)
    with open('/tmp/attn-sweep-5090.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for i,dest in enumerate(destinations):
            cmd=[sys.executable,str(dest/'run.py'),'--arch','120a','--run-only','--out',str(dest),'--adaptive-ci','1',
                 '--warmup','2','--reps','5','--iters','5']
            if i:cmd+=['--shard',f'{i-1}/2']
            subprocess.run(cmd,check=True)
    assert merge_runs(destinations[1:],root/'merged-real-shards')==0
    def rows(path):
        return list(csv.DictReader((path/'summary.csv').open()))
    arows=rows(destinations[0]);brows=rows(root/'merged-real-shards')
    assert {(r['case'],r['variant'],r['status']) for r in arows}=={(r['case'],r['variant'],r['status']) for r in brows}
    early=[r for r in arows if r['variant']=='pipe' and r.get('adaptive_stop_reason')=='CI_REACHED' and int(r['rounds_used'])<6]
    assert early, 'no stable pair reached target early; retain evidence, rerun in a quiet GPU slot'
    for r in early:print('EARLY',r['case'],r['rounds_used'],r['final_ci95_half_width_pct'],flush=True)
    # Partition original measured observations, without changing any sample or timestamp.
    # Full source dirs are preserved for audit; these are explicitly derived fixtures.
    derived=[]
    for index in range(2):
        dest=root/f'derived-shard{index}';shutil.copytree(destinations[0],dest);derived.append(dest)
        keep=cases[index::2];ids={c['id'] for c in keep}
        (dest/'cases.json').write_text(json.dumps(keep,indent=2)+'\n')
        manifest=json.loads((dest/'run-manifest.json').read_text())
        manifest.update(shard=dict(index=index,count=2),case_sha256=cases_hash(keep),
                        derived_fixture='partition of identical unsharded GPU observations, not another run')
        (dest/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for name in ('results.jsonl','pairs.jsonl'):
            selected=[json.loads(s) for s in (dest/name).read_text().splitlines() if json.loads(s)['case'] in ids]
            (dest/name).write_text(''.join(json.dumps(r)+'\n' for r in selected))
    assert merge_runs(derived,root/'merged-identical-observations')==0
    for file in ('summary.csv','summary.md'):
        assert (root/'merged-identical-observations'/file).read_bytes()==(destinations[0]/file).read_bytes(),file
    print('GPU ADAPTIVE/SHARD PASS: real shard coverage/status match; identical-observation merged CSV+Markdown byte-equal',flush=True)
if __name__=='__main__':main()
