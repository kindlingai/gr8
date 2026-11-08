#!/usr/bin/env python3
"""Explicit opt-in GPU regression: validates compute-once and wrong-stream controls.
Run with --out PATH. Builds in parallel; GPU runs share the campaign flock.
"""
import argparse
import concurrent.futures
import fcntl
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--nvcc',default='/usr/local/cuda/bin/nvcc');p.add_argument('--arch',default='120a');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    variants={'off':HERE/'tests/toy_off.cuh', 'compute_once':HERE/'negative_controls/compute_once/variant.cuh',
              'wrong_stream':HERE/'negative_controls/wrong_stream/variant.cuh'}
    def build(item):
        name,header=item
        cmd=[a.nvcc,'-std=c++17','-O3','-gencode',f'arch=compute_{a.arch},code=sm_{a.arch}',
             f'-DSWEEP_ADAPTER_HEADER="{HERE}/tests/toy_adapter.cuh"',
             f'-DSWEEP_REFERENCE_HEADER="{HERE}/tests/toy_off.cuh"',f'-DSWEEP_VARIANT_HEADER="{header}"',
             f'-DSWEEP_SAME_VARIANT={int(name=="off")}',str(HERE/'bench.cu'),'-o',str(a.out/name)]
        r=subprocess.run(cmd,capture_output=True,text=True);(a.out/f'{name}.build.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError(f'{name} build failed: '+r.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(build,variants.items()))
    with open('/tmp/attn-sweep-5090.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for name in variants:
            cmd=[str(a.out/name),'--warmup','1','--reps','3','--iters','3','--pools','3','--']
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            (a.out/f'{name}.stdout').write_text(r.stdout);(a.out/f'{name}.stderr').write_text(r.stderr)
            row=json.loads(r.stdout.strip().splitlines()[-1]);(a.out/f'{name}.json').write_text(json.dumps(row,indent=2)+'\n')
            if name=='off':assert r.returncode==0 and row['status']=='EXACT' and row['launch_valid'],row
            elif name=='compute_once':assert r.returncode==1 and row['status']=='MISMATCH' and row['timed_mismatch_bytes']>0,row
            else:assert r.returncode==1 and not row['launch_valid'],row
            assert row['input_slots']==3 and row['timed_checks']>0,row
            print(name,row['status'],'timed_mismatch_bytes',row['timed_mismatch_bytes'],'launch_valid',row['launch_valid'],flush=True)
    print('FRESH GPU CONTROLS PASS')
if __name__=='__main__':main()
