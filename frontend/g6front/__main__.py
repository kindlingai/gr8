"""gr8 front end: OpenAI/vLLM-compatible HTTP server in front of the gr8 engine (PROTOCOL.md).

python -m g6front --model-dir DIR [--engine-sock /run/g6/engine.sock] [--port 8100]
"""
import argparse
import asyncio
import logging

from aiohttp import web

from . import routes_chat, routes_completions, routes_misc
from .state import State


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog='g6front')
    ap.add_argument('--model-dir', required=True, help='dir with tokenizer.json, tokenizer_config.json, '
                    'chat_template.jinja, generation_config.json')
    ap.add_argument('--engine-sock', default='/run/g6/engine.sock')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8100)
    ap.add_argument('--served-model-name', nargs='+', default=['glm-5.3', 'glm53'],
                    help='accepted `model` values; the first is reported in responses')
    ap.add_argument('--max-model-len', type=int, default=0, help='override the engine hello value')
    ap.add_argument('--reasoning-fields', choices=['both', 'reasoning', 'reasoning_content'], default='reasoning',
                    help='field name(s) carrying reasoning text in deltas/messages (vLLM 0.30 sends only "reasoning")')
    ap.add_argument('--enable-prompt-tokens-details', action='store_true',
                    help='add usage.prompt_tokens_details.cached_tokens')
    ap.add_argument('--log-level', default='INFO')
    return ap.parse_args(argv)


def build_app(args) -> web.Application:
    app = web.Application(client_max_size=512 << 20)
    st = State(args)
    app['state'] = st
    app.router.add_post('/v1/chat/completions', routes_chat.handle)
    app.router.add_post('/v1/completions', routes_completions.handle)
    app.router.add_post('/tokenize', routes_misc.tokenize)
    app.router.add_post('/detokenize', routes_misc.detokenize)
    app.router.add_get('/v1/models', routes_misc.models)
    app.router.add_get('/health', routes_misc.health)
    app.router.add_get('/metrics', routes_misc.metrics)

    async def on_startup(app):
        await st.engine.start()

    async def on_cleanup(app):
        await st.engine.stop()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass
    # handler_cancellation: a client that disconnects from a non-streaming request cancels the engine request
    web.run_app(build_app(args), host=args.host, port=args.port, handler_cancellation=True, access_log=None,
                print=lambda *a: logging.getLogger('g6front').info(' '.join(map(str, a))))


if __name__ == '__main__':
    main()
