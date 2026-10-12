"""POST /v1/chat/completions (stream + non-stream), shaped like vLLM 0.30 with the GLM reasoning + tool parsers."""
import contextlib
import logging
import time

from aiohttp import web

from . import logprobs as L
from . import sampling
from .errors import ApiError, dumps
from .glm_parser import GlmParser
from .runner import Generation
from .state import State, new_id, open_sse, read_json, sse, usage
from .template import TemplateError, build_template_kwargs, normalize_tools

log = logging.getLogger('g6front.chat')


async def prepare(st: State, body: dict):
    st.check_model(body)
    st.require_engine()
    if body.get('echo'):
        raise ApiError('echo is not supported by the gr8 front end', param='echo')
    try:
        tools = normalize_tools(body.get('tools'))
        tool_choice = body.get('tool_choice')
        if tool_choice is None:
            tool_choice = 'auto' if tools else 'none'
        if isinstance(tool_choice, dict) or tool_choice == 'required':
            raise ApiError('named / required tool_choice is not supported by the gr8 front end', param='tool_choice')
        kw = build_template_kwargs(body.get('chat_template_kwargs'),
                                   add_generation_prompt=body.get('add_generation_prompt', True),
                                   continue_final_message=body.get('continue_final_message', False),
                                   reasoning_effort=body.get('reasoning_effort'), documents=body.get('documents'))
        if body.get('chat_template'):
            raise ApiError('custom chat_template is not supported', param='chat_template')
        text = st.template.render(body.get('messages'), tools=tools, **kw)
    except TemplateError as e:
        raise ApiError(str(e))
    prompt = await st.tok.aencode(text, add_special_tokens=bool(body.get('add_special_tokens', False)))
    params = sampling.build(body, len(prompt), st.max_model_len, st.tok.eos_ids, st.tok.gen_defaults, None,
                            st.max_logprobs, chat=True)
    span = await system_span(st, body, tools, kw, prompt)
    if span is not None:
        params['system_tokens'] = span
    return tools, tool_choice, prompt, params


async def system_span(st: State, body: dict, tools, kw: dict, prompt: list) -> int | None:
    """The prompt's system span (PROTOCOL.md `system_tokens`): its tokens up to where the first message after the
    leading system ones begins, found by rendering those messages and the tools alone. The engine's KV stores keep
    system prefixes by it. None when the template cannot render them alone."""
    lead = []
    for m in body.get('messages') or []:
        if not isinstance(m, dict) or m.get('role') not in ('system', 'developer'):
            break
        lead.append(m)
    if not lead and not tools:
        return 0
    try:
        text = st.template.render(lead, tools=tools, **dict(kw, add_generation_prompt=False, continue_final_message=False))
    except TemplateError:
        return None
    ids = await st.tok.aencode(text, add_special_tokens=bool(body.get('add_special_tokens', False)))
    n = 0
    for a, b in zip(ids, prompt):
        if a != b:
            break
        n += 1
    return n


async def handle(request: web.Request) -> web.StreamResponse:
    st: State = request.app['state']
    try:
        body = await read_json(request)
        tools, tool_choice, prompt, params = await prepare(st, body)
    except ApiError as e:
        return e.response()
    rid = new_id('chatcmpl')
    # the GLM reasoning parser forces skip_special_tokens=False (vLLM parser_engine.adjust_request)
    gen = Generation(st.engine, st.tok, rid, prompt, params, skip_special_tokens=False)
    parser = GlmParser(tools, tool_choice, include_reasoning=body.get('include_reasoning', True))
    if body.get('stream'):
        return await _stream(request, st, body, gen, parser, rid, params)
    try:
        return web.json_response(await _full(st, body, gen, parser, rid, params), dumps=dumps)
    except ApiError as e:
        return e.response()


def _reasoning_fields(st, delta: dict) -> dict:
    if 'reasoning' in delta:
        r = delta.pop('reasoning')
        for f in st.reasoning_fields:
            delta[f] = r
    return delta


async def _full(st, body, gen, parser, rid, params):
    created = int(time.time())
    async with contextlib.aclosing(gen.steps()) as steps:
        async for _ in steps:
            pass
    res = gen.res
    reasoning, content, calls = parser.parse_full(res.text)
    if not parser.include_reasoning:
        reasoning = None
    msg = {'role': 'assistant', 'content': content, 'refusal': None, 'annotations': None, 'audio': None,
           'function_call': None}
    if calls:
        msg['tool_calls'] = calls
    msg['reasoning'] = reasoning
    msg = _reasoning_fields(st, msg)
    finish = 'tool_calls' if calls else res.finish_reason
    lp = None
    if params['logprobs'] is not None:
        lp = L.chat_logprobs(res.ids, res.logprobs, params['logprobs'], st.tok,
                             body.get('return_tokens_as_token_ids', False))
    ret_ids = body.get('return_token_ids', False)
    return {
        'id': rid, 'object': 'chat.completion', 'created': created, 'model': st.model_name,
        'choices': [{'index': 0, 'message': msg, 'logprobs': lp, 'finish_reason': finish,
                     'stop_reason': res.stop_reason, 'token_ids': res.ids if ret_ids else None}],
        'service_tier': None, 'system_fingerprint': None,
        'usage': usage(res.prompt_tokens, res.completion_tokens, res.cached_tokens,
                       st.args.enable_prompt_tokens_details, True),
        'prompt_logprobs': L.prompt_logprobs(res.prompt_logprobs, gen.prompt, st.tok),
        'prompt_token_ids': gen.prompt if ret_ids else None, 'kv_transfer_params': None,
    }


async def _stream(request, st, body, gen, parser, rid, params):
    created = int(time.time())
    so = body.get('stream_options') or {}
    include_usage = bool(so.get('include_usage'))
    continuous = include_usage and bool(so.get('continuous_usage_stats'))
    ret_ids = body.get('return_token_ids', False)
    as_ids = body.get('return_tokens_as_token_ids', False)
    details = st.args.enable_prompt_tokens_details
    resp = await open_sse(request)
    res = gen.res
    base = {'id': rid, 'object': 'chat.completion.chunk', 'created': created, 'model': st.model_name}
    first, tools_streamed = True, False
    try:
        async with contextlib.aclosing(gen.steps()) as steps:
            async for step in steps:
                if first:
                    first = False
                    c = dict(base, choices=[{'index': 0, 'delta': {'role': 'assistant', 'content': ''},
                                             'logprobs': None, 'finish_reason': None}])
                    if ret_ids:
                        c['prompt_token_ids'] = gen.prompt
                    if continuous:
                        c['usage'] = usage(res.prompt_tokens, 0, None, False, False)
                    await resp.write(sse(c))
                finished = step.finish_reason is not None
                delta = parser.delta(step.text, finished=finished)
                if delta and delta.get('tool_calls'):
                    tools_streamed = True
                lp = None
                if params['logprobs'] is not None and step.ids and parser.include_reasoning:
                    lp = L.chat_logprobs(step.ids, step.logprobs, params['logprobs'], st.tok, as_ids)
                if delta is None:
                    if not finished and lp is None and not ret_ids:
                        continue
                    delta = {}
                choice = {'index': 0, 'delta': _reasoning_fields(st, delta), 'logprobs': lp, 'finish_reason': None,
                          'token_ids': step.ids if ret_ids else None}
                if finished:
                    fr = step.finish_reason
                    choice['finish_reason'] = 'tool_calls' if tools_streamed and fr == 'stop' else fr
                    choice['stop_reason'] = step.stop_reason
                c = dict(base, choices=[choice])
                if continuous:
                    c['usage'] = usage(res.prompt_tokens, res.completion_tokens, None, False, False)
                await resp.write(sse(c))
        if include_usage:
            await resp.write(sse(dict(base, choices=[], usage=usage(
                res.prompt_tokens, res.completion_tokens, res.cached_tokens, details, False))))
    except ApiError as e:
        await resp.write(e.sse())
    except (ConnectionResetError, ConnectionError):
        log.info('%s: client went away', rid)
        return resp
    await resp.write(b'data: [DONE]\n\n')
    await resp.write_eof()
    return resp
