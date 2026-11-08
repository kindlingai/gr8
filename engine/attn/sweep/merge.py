#!/usr/bin/env python3
"""Rebuild one summary from a complete set of disjoint case-shard run dirs."""
import argparse
from pathlib import Path
from run import merge_runs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('runs',type=Path,nargs='+')
    a=p.parse_args()
    try:
        status=merge_runs(a.runs,a.out)
    except (ValueError,OSError,KeyError,TypeError) as e:
        p.error(str(e))
    print(f'MERGED {a.out}/summary.md and summary.csv')
    return status
if __name__=='__main__':raise SystemExit(main())
