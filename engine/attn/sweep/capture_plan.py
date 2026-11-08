#!/usr/bin/env python3
"""Produce a bounded index-only capture plan, never launch an engine."""
import argparse
import json
from pathlib import Path


def plan(output_dir, revision, knob_manifest):
    if not isinstance(knob_manifest, dict) or not knob_manifest:
        raise ValueError('supply the actual nonempty engine knob manifest, not guessed defaults')
    if not revision:
        raise ValueError('engine revision is required')
    return dict(schema=1, output_dir=str(Path(output_dir).resolve()), engine_revision=revision,
                knob_manifest=knob_manifest, ranks=[0], layers=[2, 8, 14, 20, 21],
                prompt_lengths=[32768, 131072], chunk_positions=[0, 16384, 24576, 65536, 122880],
                max_records=40, max_bytes=768*1024*1024)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--engine-revision', required=True)
    p.add_argument('--knob-manifest', type=Path, required=True)
    a=p.parse_args()
    print(json.dumps(plan(a.output_dir, a.engine_revision, json.loads(a.knob_manifest.read_text())), indent=2))
