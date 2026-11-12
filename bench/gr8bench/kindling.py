"""bench.py kindling: one entry of a kindling.json from a bench report.

Only what the report measured is written; every other field of the entry, and the file's top level, stay as they
were. Keys keep their order, so a regenerated file diffs cleanly."""
import json
from pathlib import Path

# where a field goes when an entry does not have it yet
ORDER = ['id', 'label', 'nodes', 'decode', 'decode_prompt_tokens', 'prefill', 'concurrency',
         'concurrency_prompt_tokens', 'max_context', 'kv_pool_tokens', 'max_concurrent',
         'max_tokens_per_chat', 'max_tokens_per_chat_note', 'quality', 'notes']
DECODE = ('code', 'prose', 'structured')
NEEDLE = 'Needle recall'


def _put(d, key, value, order):
    """d with d[key] = value: an existing key keeps its place, a new one goes after the last key that comes before
    it in order (first if none does)."""
    if key in d:
        d[key] = value
        return d
    rank = {k: i for i, k in enumerate(order)}
    keys = list(d)
    at = 0
    for i, k in enumerate(keys):
        if rank.get(k, len(order)) < rank.get(key, len(order)):
            at = i + 1
    items = list(d.items())
    items.insert(at, (key, value))
    d.clear()
    d.update(items)
    return d


def _size(key):
    """'128K' -> 131072"""
    return int(key[:-1]) * 1024


def fields(rep):
    """-> ({field: value} the report measured, [what it did not])"""
    out, skipped = {}, []
    dec = rep.get('decode') or {}
    got = {w: round(dec[w]['summary']['tps'], 1) for w in DECODE if (dec.get(w) or {}).get('summary')}
    if got:
        out['decode'] = got
    skipped += [f'decode {w}' for w in DECODE if w not in got]
    pf = [{'context': _size(k), 'tps': round(v['tok_s'])} for k, v in (rep.get('prefill') or {}).items()
          if v.get('tok_s') is not None]
    if pf:
        out['prefill'] = sorted(pf, key=lambda x: x['context'])
    else:
        skipped.append('prefill')
    cc = [{'streams': int(n), 'aggregate_tps': round(v['tps'], 1)} for n, v in (rep.get('concurrency') or {}).items()
          if v.get('tps') is not None]
    if cc:
        out['concurrency'] = sorted(cc, key=lambda x: x['streams'])
    else:
        skipped.append('concurrency')
    needle = {k: v for k, v in ((rep.get('correct') or {}).get('needle') or {}).items() if 'passed' in v}
    if needle:
        keys = sorted(needle, key=_size)
        found = [k for k in keys if needle[k]['passed']]
        cut = [k for k in keys if not needle[k]['passed'] and needle[k].get('cut_off')]
        missed = [k for k in keys if not needle[k]['passed'] and not needle[k].get('cut_off')]
        out['quality'] = '; '.join(x for x in (found and 'found at ' + ', '.join(found),
                                               cut and 'found at ' + ', '.join(cut) + ' (answer cut off)',
                                               missed and 'missed at ' + ', '.join(missed)) if x)
    else:
        skipped.append('quality (needle)')
    max_len = (rep.get('info') or {}).get('max_model_len')
    if max_len:
        out['max_context'] = max_len
    else:
        skipped.append('max_context')
    return out, skipped


def update(doc, rep, cid, label=None, nodes=None, default=False):
    """Create or update the config cid of doc (a parsed kindling.json) from rep. -> (doc, filled, skipped)"""
    configs = doc.setdefault('configs', [])
    entry = next((c for c in configs if c.get('id') == cid), None)
    if entry is None:
        entry = {'id': cid}
        configs.append(entry)
    if label is not None:
        _put(entry, 'label', label, ORDER)
    if nodes is not None:
        _put(entry, 'nodes', nodes, ORDER)
    got, skipped = fields(rep)
    for k, v in got.items():
        if k == 'decode':   # a workload this report did not run keeps its old number
            _put(entry, k, {**(entry.get(k) or {}), **v}, ORDER)
        elif k == 'quality':   # only the needle line is ours
            q = [dict(x) for x in entry.get('quality') or []]
            hit = next((x for x in q if x.get('name') == NEEDLE), None)
            if hit:
                hit['value'] = v
            else:
                q.append({'name': NEEDLE, 'value': v})
            _put(entry, k, q, ORDER)
        else:
            _put(entry, k, v, ORDER)
    if default:
        _put(doc, 'default_config', cid, ['version', 'title', 'variant', 'summary', 'model', 'hardware', 'stack',
                                          'default_config', 'configs'])
    return doc, list(got), skipped


def run(report_path, out, cid, label=None, nodes=None, default=False):
    rep = json.loads(Path(report_path).read_text())
    path = Path(out)
    doc = json.loads(path.read_text()) if path.exists() else {'version': 1, 'configs': []}
    doc, filled, skipped = update(doc, rep, cid, label, nodes, default)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + '\n')
    print(f'{out}: config {cid}: wrote {", ".join(filled) or "nothing"}')
    if skipped:
        print(f'  not in the report, left as they were: {", ".join(skipped)}')
    return 0
