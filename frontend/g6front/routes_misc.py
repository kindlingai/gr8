"""/tokenize, /detokenize, /v1/models, /health, /metrics."""
from aiohttp import web

from . import metrics as M
from .engine_client import EngineError
from .errors import ApiError, dumps
from .state import State, read_json
from .template import TemplateError, build_template_kwargs, normalize_tools


async def tokenize(request: web.Request) -> web.Response:
    st: State = request.app['state']
    try:
        body = await read_json(request)
        st.check_model(body)
        if 'messages' in body:
            if body.get('continue_final_message') and body.get('add_generation_prompt', True):
                raise ApiError('Cannot set both `continue_final_message` and `add_generation_prompt` to True.')
            try:
                kw = build_template_kwargs(body.get('chat_template_kwargs'),
                                           add_generation_prompt=body.get('add_generation_prompt', True),
                                           continue_final_message=body.get('continue_final_message', False))
                text = st.template.render(body['messages'], tools=normalize_tools(body.get('tools')), **kw)
            except TemplateError as e:
                raise ApiError(str(e))
            ids = await st.tok.aencode(text, add_special_tokens=bool(body.get('add_special_tokens', False)))
        elif isinstance(body.get('prompt'), str):
            ids = await st.tok.aencode(body['prompt'], add_special_tokens=bool(body.get('add_special_tokens', True)))
        else:
            raise ApiError('`prompt` (string) or `messages` is required')
    except ApiError as e:
        return e.response()
    strs = [st.tok.id_to_token(i) for i in ids] if body.get('return_token_strs') else None
    return web.json_response({'count': len(ids), 'max_model_len': st.max_model_len, 'tokens': ids,
                              'token_strs': strs}, dumps=dumps)


async def detokenize(request: web.Request) -> web.Response:
    st: State = request.app['state']
    try:
        body = await read_json(request)
        st.check_model(body)
        toks = body.get('tokens')
        if not isinstance(toks, list) or not all(isinstance(i, int) for i in toks):
            raise ApiError('`tokens` must be a list of token ids', param='tokens')
    except ApiError as e:
        return e.response()
    return web.json_response({'prompt': st.tok.decode(toks, skip_special_tokens=False)}, dumps=dumps)


async def models(request: web.Request) -> web.Response:
    st: State = request.app['state']
    data = [{'id': name, 'object': 'model', 'created': st.started, 'owned_by': 'gr8', 'root': st.args.model_dir,
             'parent': None, 'max_model_len': st.max_model_len, 'permission': []} for name in st.served]
    return web.json_response({'object': 'list', 'data': data}, dumps=dumps)


async def health(request: web.Request) -> web.Response:
    st: State = request.app['state']
    return web.Response(status=200 if st.engine.ready else 503)


async def metrics(request: web.Request) -> web.Response:
    st: State = request.app['state']
    try:
        snap = await st.engine.metrics()
    except (EngineError, TimeoutError) as e:
        return web.Response(status=503, text=f'engine metrics unavailable: {e}\n')
    return web.Response(text=M.render(snap, st.model_name), content_type='text/plain', charset='utf-8')
