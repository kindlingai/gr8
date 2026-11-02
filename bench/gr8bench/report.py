"""The report: ASCII tables on stdout, the verdicts, and the comparison with a baseline report."""
import re

from . import stats

NOISE = 0.03   # speed gates: worse than the baseline by more than this, and significant where runs pair
WIDE = 100     # longest cell; the JSON report has the whole text
NUMBER = re.compile(r'^[-\d.,]+( \(.*\))?%?$')   # right-aligned cells: a number, maybe with its interval


def table(title, head, rows):
    rows = [['' if c is None else str(c) if len(str(c)) <= WIDE else str(c)[:WIDE - 3] + '...' for c in r] for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h)) for i, h in enumerate(head)]
    line = '+' + '+'.join('-' * (x + 2) for x in w) + '+'
    fmt = lambda cells: '| ' + ' | '.join(c.rjust(x) if NUMBER.match(c) else c.ljust(x) for c, x in zip(cells, w)) + ' |'
    return '\n'.join([title, line, fmt([str(h) for h in head]), line.replace('-', '=')] + [fmt(r) for r in rows] +
                     [line, ''])


def f1(x, d=1):
    return None if x is None else f'{x:,.{d}f}'


def ratio(r):
    if not r:
        return None
    s = f"{r['ratio']:.3f}"
    if r.get('ci'):
        s += f" ({r['ci'][0]:.3f}-{r['ci'][1]:.3f})"
    if r.get('p') is not None:
        s += f" p={r['p']:.3f}"
    return s


def compare(cur, base):
    """Speed against a baseline report: decode paired run by run (Holm across workloads), concurrency and prefill as
    ratios of the two runs' numbers. -> {section: {key: ratio dict}}"""
    out = {'decode': {}, 'concurrency': {}, 'prefill': {}}
    if not base:
        return out
    cd, bd = cur.get('decode') or {}, base.get('decode') or {}
    keys = [w for w in cd if w in bd]
    pr = {w: stats.paired(bd[w]['runs'], cd[w]['runs']) for w in keys}
    live = [w for w in keys if pr[w]]
    for w, p in zip(live, stats.holm([pr[w]['p'] for w in live])):
        out['decode'][w] = {**pr[w], 'p': p}
    for sec, k in (('concurrency', 'tps'), ('prefill', 'tok_s')):
        for key, v in (cur.get(sec) or {}).items():
            b = (base.get(sec) or {}).get(key) or {}
            if v.get(k) and b.get(k):
                out[sec][key] = {'ratio': v[k] / b[k]}
    return out


def verdicts(rep, cmp, kl_max, base=None):
    """-> list of (check, verdict, detail). Verdicts: PASS, FAIL, WARN, INFO."""
    out = []
    add = lambda c, v, d='': out.append((c, v, d))
    s = rep.get('smoke')
    if s:
        add('smoke', 'PASS' if s['cases'] and s['passed'] == s['cases'] else 'FAIL',
            f"{s['passed']}/{s['cases']}" + (f" failed: {', '.join(f.split(':')[0] for f in s['failed'])}" if s['failed'] else ''))
    c = rep.get('correct') or {}
    if 't0' in c:
        bh = (((base or {}).get('correct') or {}).get('t0') or {}).get('hash')
        add('T=0 replies', 'INFO', c['t0']['hash'] + ('' if not bh else ' (equal to baseline)' if bh == c['t0']['hash']
                                                   else f' (baseline {bh})'))
    if 'count' in c:
        add('count to 200', 'PASS' if c['count']['clean'] == c['count']['runs'] else 'FAIL',
            f"{c['count']['clean']}/{c['count']['runs']} clean")
    if 'cached' in c:
        n = c['cached']['second']
        add('prompt cache report', 'PASS' if n else 'FAIL',
            f"{n} of {c['cached']['prompt_tokens']} cached" if n is not None else 'no cached_tokens in usage')
    if 'kl' in c:
        k = c['kl']
        if 'mean' in k:
            add('KL against reference', 'PASS' if kl_max is None or k['mean'] <= kl_max else 'FAIL',
                f"mean {k['mean']:.5f}, p95 {k['p95']:.4f}, top-1 {100 * k['top1']:.1f}%")
        else:
            add('KL against reference', 'INFO', k.get('skipped', ''))
    for size, n in (c.get('needle') or {}).items():
        if 'skipped' in n:
            add(f'needle {size}', 'INFO', n['skipped'])
        else:
            add(f'needle {size}', 'PASS' if n['passed'] else 'FAIL', f"{n['prompt_tokens']:,} tokens"
                + (', found in reasoning only' if n.get('in_reasoning_only') else ''))
    a = rep.get('agentic') or {}
    if 'toolcall' in a:
        t = a['toolcall']
        add('tool call at 42K', 'WARN' if t.get('error') or t['divergent'] else 'PASS',
            t.get('error') or f"{t['divergent']} of {t['runs']} diverged, {t['distinct']} distinct")
    if 'agent_tools' in a:
        t = a['agent_tools']
        add('agent tools', 'FAIL' if t.get('error') or t['passed'] < t['runs'] else 'PASS',
            t.get('error') or f"{t['passed']}/{t['runs']}, {t['refused']} refused, {t['bad_args']} bad args")
    if 'longreply' in a:
        t = a['longreply']
        add('long reply, 16K', 'FAIL' if t.get('error') or t['verdict'] == 'FAIL' else
            'PASS' if t['verdict'] == 'PASS' else 'WARN', t.get('error') or '; '.join(t['why']) or t['verdict'])
    q = rep.get('quality') or {}
    if 'gsm8k' in q:
        g = q['gsm8k']
        add('GSM8K', 'INFO', g.get('error') or f"{g['correct']}/{g['n']} = {100 * g['accuracy']:.1f}%")
    if 'humaneval' in q:
        h = q['humaneval']
        add('HumanEval', 'INFO', h.get('error') or h.get('note') or f"{h['passed']}/{h['n']} = {100 * h['pass_at_1']:.1f}%")
    add('suspect characters', 'PASS' if not rep.get('suspect') else 'FAIL', f"{len(rep.get('suspect') or [])} replies")
    for st, h in (rep.get('heat') or {}).items():
        if h.get('guard'):
            add(f'{st}: heat guard', 'WARN', 'numbers affected: ' + ', '.join(
                f"{t['host']} held at {t['temp']:.0f} C" for t in h['guard']))
    for w, d in (rep.get('decode') or {}).items():
        if d['problems']:
            add(f'decode {w}', 'FAIL', '; '.join(d['problems'][:3]))
        r = cmp['decode'].get(w)
        if r:
            slower = r['ratio'] < 1 - NOISE and r['p'] < 0.05
            add(f'decode {w} vs baseline', 'FAIL' if slower else 'PASS', ratio(r))
    for n, d in (rep.get('concurrency') or {}).items():
        if d['failed']:
            add(f'{n} streams', 'FAIL', f"{d['failed']} streams failed")
        r = cmp['concurrency'].get(n)
        if r:
            add(f'{n} streams vs baseline', 'WARN' if r['ratio'] < 1 - NOISE else 'PASS', ratio(r))
    for k, r in cmp['prefill'].items():
        add(f'prefill {k} vs baseline', 'FAIL' if r['ratio'] < 1 - NOISE else 'PASS', ratio(r))
    return out


def render(rep, cmp, checks):
    out = []
    m = rep['meta']
    out.append(table('Run', ['setting', 'value'], [[k, v] for k, v in (
        ('model', m['model']), ('level', m['level']), ('label', m['label']), ('url', m['url']),
        ('hosts', ', '.join(m['hosts']) or 'none (no clock control)'),
        ('stage clocks', m['stage_clocks']), ('cooldown', m['cooldown']),
        ('started', m['started']), ('took', m.get('took')), ('baseline', m.get('baseline') or 'none'))]))
    out.append(table('Checks', ['check', 'verdict', 'detail'], [list(c) for c in checks]))
    dec = rep.get('decode') or {}
    if dec:
        rows = []
        for w, d in dec.items():
            s = d['summary'] or {}
            ci = s.get('ci') or [None, None]
            rows.append([w, f"{f1(s.get('tps'))} ({f1(ci[0])}-{f1(ci[1])})" if ci[0] else f1(s.get('tps')),
                         f"{100 * s['cv']:.0f}%" if s.get('cv') is not None else None, f1(s.get('tok_per_step'), 2),
                         f1(s.get('ms_per_step')), f1(s.get('accept'), 2), f1(s.get('ttft_ms'), 0),
                         f"{len(d['runs'])}", ratio(cmp['decode'].get(w))])
        out.append(table('Decode, one stream', ['workload', 'tok/s (95% CI)', 'spread', 'tok/step', 'ms/step',
                                                 'accept', 'TTFT ms', 'runs', 'vs baseline'], rows))
    con = rep.get('concurrency') or {}
    if con:
        rows = [[n, f1(d['tps']), f1(d['per_stream']), f1(d['ttft_p50'] and 1000 * d['ttft_p50'], 0),
                 f1(d['ttft_p90'] and 1000 * d['ttft_p90'], 0), f1(d['tok_per_step'], 2), d['failed'] or '',
                 ratio(cmp['concurrency'].get(n))] for n, d in con.items()]
        out.append(table('Concurrent streams (mixed prompts)', ['streams', 'total tok/s', 'per stream', 'TTFT p50 ms',
                                                                'TTFT p90 ms', 'tok/step', 'failed', 'vs baseline'], rows))
    pf = rep.get('prefill') or {}
    if pf:
        rows = []
        for k, d in pf.items():
            if 'skipped' in d:
                rows.append([k, d['skipped'], None, None, None, None, None, None])
                continue
            rows.append([k, f"{d['prompt_tokens']:,}", f1(d['ttft_s'], 2), f1(d['tok_s'], 0) + (' (not cold)' if
                         d['not_cold'] else ''), f1(d['decode_tps']), f1(d['ttft_cached_s'], 2), d['cached_tokens'],
                         ratio(cmp['prefill'].get(k))])
        out.append(table('Prefill (cold, salted prompt)', ['prompt', 'tokens', 'TTFT s', 'prefill tok/s',
                                                           'decode after', 'TTFT cached s', 'cached', 'vs baseline'],
                         rows))
    heat = rep.get('heat') or {}
    if heat:
        rows = []
        for st, h in heat.items():
            first = True
            for box, b in h['boxes'].items():
                rows.append([st if first else '', f"{h['cooldown']['seconds']:.0f}" if first and h.get('cooldown')
                             else '', box, f1(b['temp_start'], 0), f1(b['temp_max'], 0),
                             f"{b['sm_mean']:.0f} / {b['sm_min']:.0f}" + (f" (cap {b['cap_mhz']})" if b['cap_mhz'] else ''),
                             f"{100 * b['under_frac']:.0f}%", f1(b['power_mean']), ', '.join(b['throttle'])])
                first = False
        out.append(table('Heat per stage', ['stage', 'cooldown s', 'box', 'start C', 'max C', 'SM mean / min MHz',
                                            'under 2000', 'power W', 'throttle'], rows))
    worst = [c for c in checks if c[1] == 'FAIL']
    if worst:
        out.append('RESULT: FAIL (' + ', '.join(c[0] for c in worst) + ')')
    elif m.get('interrupted'):
        out.append('RESULT: INCOMPLETE (interrupted after ' + ', '.join(rep.get('heat') or ['setup']) + ')')
    else:
        out.append('RESULT: PASS')
    return '\n'.join(out)
