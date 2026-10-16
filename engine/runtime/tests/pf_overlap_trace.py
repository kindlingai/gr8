#!/usr/bin/env python3
"""Local-clock, lane-aware analysis. No cross-rank timestamp subtraction.
Usage: analyze_overlap.py RANK0_STDERR RANK1_STDERR [EXPECTED_RECORDS_PER_RANK]
PFCOMM stamps bracket staged peer wait directly; put->mHC-entry includes other-lane work.
"""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    lane=0
    comm={}
    pxt={}
    for line in Path(path).read_text().splitlines():
        if line == '# PXT second lane':
            lane=1
        elif line.startswith('PFCOMM '):
            _,s,c,a,b,d=line.split()
            key=(lane,int(s),int(c))
            assert key not in comm,('duplicate COMM',key)
            comm[key]=tuple(map(int,(a,b,d)))
        elif line.startswith('PXT '):
            _,s,c,m,prod,put,entry,wmin,wmax,end=line.split()
            key=(lane,int(s),int(c))
            assert key not in pxt,('duplicate PXT',key)
            pxt[key]=dict(zip(('rows','prod','put','entry','wmin','wmax','end'),map(int,(m,prod,put,entry,wmin,wmax,end))))
    assert set(comm)==set(pxt),('coverage mismatch',len(comm),len(pxt))
    for k,(a,b,d) in comm.items():
        assert 0<a<=b<=d<=pxt[k]['entry']<=pxt[k]['end'],(k,comm[k],pxt[k])
    return comm,pxt


def summary(comm,pxt):
    wait=[(b-a)/1e6 for a,b,d in comm.values()]
    copy=[(d-b)/1e6 for a,b,d in comm.values()]
    return dict(records=len(comm),rows=sum(x['rows'] for x in pxt.values()),
                exposed_wait_s=sum(wait)/1000,copy_s=sum(copy)/1000,
                wait_p50_ms=statistics.median(wait),wait_max_ms=max(wait),
                mhc_s=sum((x['end']-x['entry'])/1e9 for x in pxt.values()))


def main():
    expected=int(sys.argv[3]) if len(sys.argv)>3 else 360
    out={}
    for rank,path in enumerate(sys.argv[1:3]):
        comm,pxt=load(path)
        assert len(comm)==expected,(rank,len(comm),expected)
        out[f'rank{rank}']=summary(comm,pxt)
        out[f'rank{rank}']['lanes']={str(l):summary({k:v for k,v in comm.items() if k[0]==l},
                                                  {k:v for k,v in pxt.items() if k[0]==l}) for l in sorted({k[0] for k in comm})}
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
