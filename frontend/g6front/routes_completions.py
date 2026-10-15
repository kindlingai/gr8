"""POST /v1/completions: text or token-id prompt, logprobs, prompt_logprobs (vLLM 0.30 shapes)."""
import contextlib
import logging
import time

from aiohttp import web

from . import logprobs as L
from . import sampling
from .errors import ApiError, dumps
from .runner import Generation
from .state import State, new_id, open_sse, read_json, sse, usage

log = logging.getLogger('g6front.completions')


async def _prompt_ids(st: State, body: dict) -> list[int]:
    p = body.get('prompt')
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], (str, list)):
        p = p[0]  # a batch of one
    if isinstance(p, str):
        return await st.tok.aencode(p, add_special_tokens=bool(body.get('add_special_tokens', True)))
    if isinstance(p, list) and p and all(isinstance(i, int) and not isinstance(i, bool) for i in p):
        if any(i < 0 or i >= st.tok.vocab_size for i in p):
            raise ApiError('prompt contains a token id out of vocabulary range', param='prompt')
        return p
    if isinstance(p, list) and len(p) > 1:
        raise ApiError('batched prompts are not supported by the gr8 front end; send one prompt per request',
                       param='prompt')
    raise ApiError('`prompt` must be a string or a non-empty list of token ids', param='prompt')


async def handle(request: web.Request) -> web.StreamResponse:
    st: State = request.app['state']
    try:
        body = await read_json(request)
        st.check_model(body)
        st.require_engine()
        for k in ('echo', 'suffix'):
            if body.get(k):
                raise ApiError(f'`{k}` is not supported by the gr8 front end', param=k)
        prompt = await _prompt_ids(st, body)
        params = sampling.build(body, len(prompt), st.max_model_len, st.tok.eos_ids, st.tok.gen_defaults, 16,
                                st.max_logprobs, chat=False)
    except ApiError as e:
        return e.response()
    rid = new_id('cmpl')
    gen = Generation(st.engine, st.tok, rid, prompt, params,
                     skip_special_tokens=body.get('skip_special_tokens', True))
    if body.get('stream'):
        return await _stream(request, st, body, gen, rid, params)
    try:
        return web.json_response(await _full(st, body, gen, rid, params), dumps=dumps)
    except ApiError as e:
        return e.response()


async def _full(st, body, gen, rid, params):
    created = int(time.time())
    async with contextlib.aclosing(gen.steps()) as steps:
        async for _ in steps:
            pass
    res = gen.res
    lp = None
    if params['logprobs'] is not None:
        lp = L.completion_logprobs(res.ids, res.logprobs, params['logprobs'], st.tok,
                                   body.get('return_tokens_as_token_ids', False))
    ret_ids = body.get('return_token_ids', False)
    return {
        'id': rid, 'object': 'text_completion', 'created': created, 'model': st.model_name,
        'choices': [{'index': 0, 'text': res.text, 'logprobs': lp, 'finish_reason': res.finish_reason,
                     'stop_reason': res.stop_reason, 'token_ids': res.ids if ret_ids else None,
                     'prompt_logprobs': L.prompt_logprobs(res.prompt_logprobs, gen.prompt, st.tok),
                     'prompt_token_ids': gen.prompt if ret_ids else None}],
        'service_tier': None, 'system_fingerprint': None,
        'usage': usage(res.prompt_tokens, res.completion_tokens, res.cached_tokens,
                       st.args.enable_prompt_tokens_details, True),
        'kv_transfer_params': None,
    }


async def _stream(request, st, body, gen, rid, params):
    created = int(time.time())
    so = body.get('stream_options') or {}
    include_usage = bool(so.get('include_usage'))
    continuous = include_usage and bool(so.get('continuous_usage_stats'))
    as_ids = body.get('return_tokens_as_token_ids', False)
    ret_ids = body.get('return_token_ids', False)
    resp = await open_sse(request)
    res = gen.res
    base = {'id': rid, 'object': 'text_completion', 'created': created, 'model': st.model_name}
    offset, first = 0, True
    try:
        async with contextlib.aclosing(gen.steps()) as steps:
            async for step in steps:
                finished = step.finish_reason is not None
                lp = None
                if params['logprobs'] is not None and step.ids:
                    lp = L.completion_logprobs(step.ids, step.logprobs, params['logprobs'], st.tok, as_ids, offset)
                    offset = lp['text_offset'][-1] + len(lp['tokens'][-1])
                if not step.text and not finished and lp is None:
                    continue
                choice = {'index': 0, 'text': step.text, 'logprobs': lp, 'finish_reason': step.finish_reason,
                          'stop_reason': step.stop_reason}
                if ret_ids:
                    choice['token_ids'] = step.ids
                    if first:
                        choice['prompt_token_ids'] = gen.prompt
                if first and res.prompt_logprobs is not None:
                    choice['prompt_logprobs'] = L.prompt_logprobs(res.prompt_logprobs, gen.prompt, st.tok)
                first = False
                c = dict(base, choices=[choice])
                if continuous:
                    c['usage'] = usage(res.prompt_tokens, res.completion_tokens, None, False, False)
                await resp.write(sse(c))
        if include_usage:
            await resp.write(sse(dict(base, choices=[], usage=usage(
                res.prompt_tokens, res.completion_tokens, res.cached_tokens,
                st.args.enable_prompt_tokens_details, False))))
    except ApiError as e:
        await resp.write(e.sse())
    except (ConnectionResetError, ConnectionError):
        log.info('%s: client went away', rid)
        return resp
    await resp.write(b'data: [DONE]\n\n')
    await resp.write_eof()
    return resp
