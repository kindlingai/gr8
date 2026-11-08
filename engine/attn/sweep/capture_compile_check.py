#!/usr/bin/env python3
"""Compile changed engine TUs off-ring; no GPU access, linking, or model boot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--nr', type=int, default=4, choices=(3,4,6))
    p.add_argument('--jobs', type=int, default=2)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    def compile_one(job):
        src,enabled=job
        name=f'{Path(src).stem}-tp{a.nr}-capture{enabled}'
        cmd=[a.nvcc,'-std=c++17','-O0','-gencode=arch=compute_121a,code=sm_121a',
             '-DG6_MODEL_DS41=1',f'-DG6_NR={a.nr}','-DDEC_S2','-DG6_MX=48','-DG6_CP=1','-DG6_RS_LOADER']
        if enabled:cmd+=['-DGR8_ATTN_SWEEP_CAPTURE=1']
        cmd+=['-c',str(ROOT/src),'-o',str(a.out/(name+'.o'))]
        (a.out/(name+'.cmd')).write_text(' '.join(cmd)+'\n')
        with (a.out/(name+'.log')).open('w') as f:
            status=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        print(f'{name}: exit={status}',flush=True)
        if status:return status
        symbols=subprocess.check_output(['nm','-C',str(a.out/(name+'.o'))],text=True)
        has_hook='ds41_sweep_capture_indices' in symbols
        if has_hook!=bool(enabled):
            print(f'{name}: unexpected hook symbols: {has_hook}',flush=True)
            return 1
        print(f'{name}: hook_symbols={has_hook} EXPECTED',flush=True)
        return 0
    jobs=[(src,enabled) for src in ('engine/attn/ds41_attn.cu','engine/runtime/g6m4.cu') for enabled in (0,1)]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        return int(any(list(pool.map(compile_one,jobs))))

if __name__=='__main__':raise SystemExit(main())
