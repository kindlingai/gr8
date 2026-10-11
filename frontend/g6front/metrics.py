"""Engine `metrics` snapshot -> Prometheus text with vLLM's metric names (kept only so the old rulers work)."""


def render(m: dict, model_name: str) -> str:
    lab = f'engine="0",model_name="{model_name}"'
    out = []

    def metric(name, kind, help_, samples):
        out.append(f'# HELP {name} {help_}')
        out.append(f'# TYPE {name} {kind}')
        for extra, v in samples:
            labels = lab + (',' + extra if extra else '')
            out.append(f'{name}{{{labels}}} {float(v)}')

    spec = m.get('spec') or {}
    pc = m.get('prefix_cache') or {}
    rq = m.get('requests') or {}
    tk = m.get('tokens') or {}
    metric('vllm:num_requests_running', 'gauge', 'Number of requests in model execution batches.',
           [('', rq.get('running', 0))])
    metric('vllm:num_requests_waiting', 'gauge', 'Number of requests waiting to be processed.',
           [('', rq.get('waiting', 0))])
    metric('vllm:prefix_cache_queries_total', 'counter', 'Prefix cache queries, in terms of number of queried tokens.',
           [('', pc.get('queries', 0))])
    metric('vllm:prefix_cache_hits_total', 'counter', 'Prefix cache hits, in terms of number of cached tokens.',
           [('', pc.get('hits', 0))])
    metric('vllm:prompt_tokens_total', 'counter', 'Number of prefill tokens processed.', [('', tk.get('prompt', 0))])
    metric('vllm:generation_tokens_total', 'counter', 'Number of generation tokens processed.',
           [('', tk.get('generation', 0))])
    fin = rq.get('finished') or {}
    metric('vllm:request_success_total', 'counter', 'Count of successfully processed requests.',
           [(f'finished_reason="{r}"', fin.get(r, 0)) for r in ('stop', 'length', 'abort', 'error')])
    metric('vllm:spec_decode_num_drafts_total', 'counter', 'Number of spec decoding drafts.',
           [('', spec.get('drafts', 0))])
    metric('vllm:spec_decode_num_draft_tokens_total', 'counter', 'Number of draft tokens.',
           [('', spec.get('draft_tokens', 0))])
    metric('vllm:spec_decode_num_accepted_tokens_total', 'counter', 'Number of accepted tokens.',
           [('', spec.get('accepted', 0))])
    metric('vllm:spec_decode_num_accepted_tokens_per_pos_total', 'counter', 'Accepted tokens per draft position.',
           [(f'position="{i}"', v) for i, v in enumerate(spec.get('accepted_per_pos') or [])])
    # S2-only native allocator/scheduler snapshot; legacy exposition is unchanged.
    s2 = m.get('scheduler')
    if s2 is not None:
        for key in ('resident', 'queued', 'retiring', 'reserved_pages', 'free_pages', 'largest_free_run', 'held_pages',
                    'worst_pages'):
            if key in s2:
                metric('g6:s2_' + key, 'gauge', 'S2 native ' + key.replace('_', ' ') + '.', [('', s2[key])])
        for key in ('admission_stalls', 'fragmentation_stalls', 'grown', 'stalled_rounds'):
            if key in s2:
                metric('g6:s2_' + key + '_total', 'counter', 'S2 native ' + key.replace('_', ' ') + '.', [('', s2[key])])
    return '\n'.join(out) + '\n'
