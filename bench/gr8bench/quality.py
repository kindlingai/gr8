"""GSM8K and HumanEval pass@1, greedy. The method follows the glm-5.3-flash-gx10 recipe (experimental/quality).

The data comes from OpenAI's repositories on first use and stays in the data directory. HumanEval runs the model's
programs, each in a fresh process with no network: under `unshare -rn` where the kernel allows it, else in a
python:3.12-slim container with --network none. With neither, the programs are written but not scored."""
import concurrent.futures as cf
import gzip
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

GSM8K_URL = 'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl'
HUMANEVAL_URL = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
PRELUDE = 'from typing import *\nimport math, re, string, collections, itertools, functools, heapq, bisect\n'
IMAGE = 'python:3.12-slim'


def fetch(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        path.write_bytes(gzip.decompress(data) if url.endswith('.gz') else data)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k(cl, data_dir, n, effort, workers=8):
    rows = fetch(GSM8K_URL, Path(data_dir) / 'gsm8k_test.jsonl')[:n]

    def one(q):
        r = cl.chat([{'role': 'user', 'content': 'Solve the problem. Show brief working, then give the final answer '
                      "on the last line as '#### <number>'.\n\n" + q['question']}], 4096, effort=effort)
        txt = r['content']
        m = re.findall(r'####\s*([-\d.,$]+)', txt) or re.findall(r'(-?[\d,]*\.?\d+)', txt)
        got, gold = (_num(m[-1]) if m else None), _num(q['answer'].split('####')[-1])
        return got is not None and gold is not None and abs(got - gold) < 1e-6
    with cf.ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(one, rows))
    return {'n': len(rows), 'correct': sum(ok), 'accuracy': sum(ok) / len(rows)}


def _sandbox():
    """-> argv prefix that runs `python3 FILE` with no network, or None."""
    if shutil.which('unshare'):
        try:
            if subprocess.run(['unshare', '-rn', 'true'], capture_output=True, timeout=10).returncode == 0:
                return ['unshare', '-rn']
        except subprocess.TimeoutExpired:
            pass
    return None


def _docker():
    for d in (['docker'], ['sudo', '-n', 'docker']):
        try:
            if shutil.which(d[-1]) and subprocess.run(d + ['info'], capture_output=True, timeout=30).returncode == 0:
                return d
        except subprocess.TimeoutExpired:
            pass
    return None


def humaneval(cl, data_dir, effort, workers=8):
    rows = fetch(HUMANEVAL_URL, Path(data_dir) / 'HumanEval.jsonl')
    work = Path(tempfile.mkdtemp(prefix='gr8bench-he-'))

    def one(p):
        r = cl.chat([{'role': 'user', 'content': 'Complete the following Python function. Reply with the whole '
                      'function in one ```python code block and nothing else.\n\n' + p['prompt']}], 4096, effort=effort)
        txt = r['content']
        m = re.findall(r'```(?:python)?\n(.*?)```', txt, re.S)
        code = m[0] if m else txt
        if f"def {p['entry_point']}" not in code:
            code = p['prompt'] + code
        (work / (p['task_id'].replace('/', '_') + '.py')).write_text(
            PRELUDE + code + '\n\n' + p['test'] + f"\n\ncheck({p['entry_point']})\n")
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, rows))
    files = sorted(work.glob('*.py'))
    prefix = _sandbox()
    if prefix:
        passed = 0
        for f in files:
            try:
                passed += subprocess.run(prefix + ['python3', '-I', str(f)], capture_output=True, timeout=15,
                                         cwd=work).returncode == 0
            except subprocess.TimeoutExpired:
                pass
        how = 'unshare -rn'
    elif (docker := _docker()):
        script = 'import glob,subprocess,sys\nok=0\nfor f in sorted(glob.glob("/q/*.py")):\n    try:\n' \
                 '        ok+=subprocess.run([sys.executable,"-I",f],capture_output=True,timeout=15).returncode==0\n' \
                 '    except subprocess.TimeoutExpired:\n        pass\nprint(ok)\n'
        p = subprocess.run(docker + ['run', '--rm', '--network', 'none', '-v', f'{work}:/q:ro', IMAGE, 'python3', '-c',
                                     script], capture_output=True, text=True, timeout=3600)
        passed = int(p.stdout.strip().splitlines()[-1]) if p.returncode == 0 and p.stdout.strip() else None
        how = f'{IMAGE}, --network none'
    else:
        return {'n': len(files), 'passed': None, 'note': f'not scored: no network sandbox; programs in {work}'}
    shutil.rmtree(work, ignore_errors=True)
    return {'n': len(files), 'passed': passed, 'pass_at_1': passed / len(files) if passed is not None else None,
            'sandbox': how}
